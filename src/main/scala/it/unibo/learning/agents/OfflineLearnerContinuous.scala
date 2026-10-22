package it.unibo.learning.agents

import ai.djl.ndarray.{NDList, NDManager}
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, Step, Trajectory}
import it.unibo.learning.network.NeuralNetworkRL.SpatialTemporal
import it.unibo.learning.network.TorchNetwork
import it.unibo.learning.network.TorchNetwork._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class OfflineLearnerContinuous(
    min: Double,
    max: Double,
    randomKey: Double,
    weight: Double,
    val stateDim: Int = SpatialTemporal.FeatureDim,
    val hiddenSize: Int = 64,
    val gamma: Float = 0.9f,
    val lr: Float = 1e-3f,
    val batchSize: Int = 64,
    val steps: Int = 160,
    val explorationSteps: Int = 10,
    val learningStepAfter: Int = 3
) extends Learner {

  private var random: Random = new Random((randomKey.toLong * 31L) ^ java.lang.Double.doubleToLongBits(weight))
  private val rootManager: NDManager = NDManager.newBaseManager()
  private val actionScaler: TorchNetwork.MinMaxActionScaler = TorchNetwork.MinMaxActionScaler(min, max)

  private val baseSeed: Int = (randomKey.toInt * 1009) ^ weight.hashCode()
  private val actor: TorchNetwork =
    TorchNetwork.createMLP(
      "actor",
      stateDim,
      Seq(hiddenSize, hiddenSize),
      1,
      Some(Activation.tanhBlock()),
      lr = lr * 0.5f,
      seed = baseSeed + 1
    )
  private val targetActor: TorchNetwork =
    TorchNetwork.createMLP(
      "target-actor",
      stateDim,
      Seq(hiddenSize, hiddenSize),
      1,
      Some(Activation.tanhBlock()),
      lr = lr * 0.5f,
      seed = baseSeed + 2
    )
  targetActor.softUpdateFrom(actor, 1.0f)

  private val critic1: TorchNetwork =
    TorchNetwork.createMLP("critic-1", stateDim + 1, Seq(hiddenSize, hiddenSize), 1, lr = lr, seed = baseSeed + 4)
  private val targetCritic1: TorchNetwork =
    TorchNetwork.createMLP(
      "target-critic-1",
      stateDim + 1,
      Seq(hiddenSize, hiddenSize),
      1,
      lr = lr,
      seed = baseSeed + 5
    )
  targetCritic1.softUpdateFrom(critic1, 1.0f)

  private val critic2: TorchNetwork =
    TorchNetwork.createMLP("critic-2", stateDim + 1, Seq(hiddenSize, hiddenSize), 1, lr = lr, seed = baseSeed + 6)
  private val targetCritic2: TorchNetwork =
    TorchNetwork.createMLP(
      "target-critic-2",
      stateDim + 1,
      Seq(hiddenSize, hiddenSize),
      1,
      lr = lr,
      seed = baseSeed + 7
    )
  targetCritic2.softUpdateFrom(critic2, 1.0f)

  private val epsilon: Double = 0.99
  private val targetUpdateTau: Float = 0.02f
  private val counterfactualWeights: Array[Double] = Array(0.5, 0.8, 0.95, 0.99, 1.0)

  private var ticks: Int = 0
  private val unstableTransitions: ArrayBuffer[OfflineLearnerContinuous.ContinuousTransition] = ArrayBuffer.empty
  private val stableTransitions: ArrayBuffer[OfflineLearnerContinuous.ContinuousTransition] = ArrayBuffer.empty
  @volatile private var fastActorEval: Array[Float] => Array[Float] = actor.snapshotFastMLP(useTanh = true)

  private def encode(state: AgentState, targetWeight: Double = weight): Array[Float] =
    SpatialTemporal.encode(state, targetWeight, max)

  def linearScaler(start: Double, end: Double, totalSteps: Int, current: Int): Double =
    if (current >= totalSteps) end
    else start + (end - start) * current.toDouble / totalSteps.toDouble

  override def policyForWeight(customWeight: Double): AgentState => Double = { state =>
    val choice = random.nextDouble()
    if (choice < linearScaler(epsilon, 0.0, explorationSteps, ticks)) {
      min + random.nextDouble() * (max - min)
    } else {
      val scaledAction = fastActorEval(encode(state, customWeight))(0).toDouble
      actionScaler.unscale(scaledAction)
    }
  }

  override def policy: AgentState => Double = policyForWeight(weight)

  override def injectRandom(random: Random): Unit = this.random = random

  override def endEpisode(trajectories: Seq[Trajectory]): Unit = {
    appendEpisodeTrajectories(trajectories)
    if (ticks <= explorationSteps && (unstableTransitions.nonEmpty || stableTransitions.nonEmpty)) {
      val nSteps = if (ticks < explorationSteps) steps else steps * learningStepAfter
      fit(nSteps)
      fastActorEval = actor.snapshotFastMLP(useTanh = true)
      if (ticks == explorationSteps && math.abs(randomKey - 2.0) < 1e-6) {
        exportEmbeddingsAndSensitivity()
      }
    }
    ticks += 1
  }

  private def exportEmbeddingsAndSensitivity(): Unit = {
    val outDir = new java.io.File("build/exports/embeddings")
    outDir.mkdirs()
    val allSamples = (unstableTransitions.take(400) ++ stableTransitions.take(400)).toVector
    if (allSamples.isEmpty) return

    withSubManager(rootManager) { sub =>
      val statesArr = allSamples.map(_.state).toArray
      val statesTensor = sub.create(statesArr)
      val hiddenTensor = actor.extractHiddenEmbedding(statesTensor)
      val predScaled = actor.evaluate(statesTensor)
      val qTensor = critic1.evaluate(statesTensor.concat(predScaled, 1))

      val nSamples = allSamples.size
      val hiddenDim = hiddenTensor.getShape.get(1).toInt

      // Compute finite-difference feature sensitivity |d(Delta t) / d(x_k)| and permutation impact for each feature k
      val eps = 0.1f
      val baseDt = (0 until nSamples).map(i => actionScaler.unscale(predScaled.getFloat(i.toLong, 0L))).toArray
      val sensitivities = Array.ofDim[Double](nSamples, stateDim)

      (0 until stateDim).foreach { k =>
        val perturbedPlus = statesArr.map { row =>
          val c = row.clone()
          c(k) = c(k) + eps
          c
        }
        val perturbedMinus = statesArr.map { row =>
          val c = row.clone()
          c(k) = c(k) - eps
          c
        }
        val outPlus = actor.evaluate(sub.create(perturbedPlus))
        val outMinus = actor.evaluate(sub.create(perturbedMinus))
        (0 until nSamples).foreach { i =>
          val dtPlus = actionScaler.unscale(outPlus.getFloat(i.toLong, 0L))
          val dtMinus = actionScaler.unscale(outMinus.getFloat(i.toLong, 0L))
          sensitivities(i)(k) = math.abs(dtPlus - dtMinus) / (2.0 * eps)
        }
      }

      val embFile = new java.io.PrintWriter(new java.io.File(outDir, s"embeddings_weight-$weight.csv"))
      try {
        val featNames = Seq(
          "deltaLatest", "deltaPrev", "selfDeltaMag", "maxNeighDeltaMag", "neighChangedRatio", "bfResidual",
          "childrenRatio", "minNeighAction", "weight"
        )
        val sensNames = featNames.map(n => s"sens_$n")
        val hidNames = (0 until hiddenDim).map(h => s"h_$h")
        embFile.println((featNames ++ Seq("predDt", "qValue", "regime") ++ sensNames ++ hidNames).mkString(","))

        (0 until nSamples).foreach { i =>
          val s = statesArr(i)
          val dt = baseDt(i)
          val qv = qTensor.getFloat(i.toLong, 0L)
          val selfActive = math.abs(s(0)) > 0.01f || math.abs(s(1)) > 0.01f || s(2) > 0.05f
          val neighActive = s(3) > 0.02f || s(4) > 0.01f || s(5) > 0.05f
          val isBackbone = s(6) > 0.15f
          val regime =
            if (selfActive && isBackbone) "Backbone_Wave"
            else if (selfActive) "Leaf_Wave"
            else if (neighActive) "Neighbor_EarlyWarning"
            else "Steady_State"
          val hVec = (0 until hiddenDim).map(h => f"${hiddenTensor.getFloat(i.toLong, h.toLong)}%.5f")
          val sensVec = sensitivities(i).map(v => f"$v%.5f")
          val sVec = s.map(v => f"$v%.5f")
          embFile.println((sVec ++ Seq(f"$dt%.5f", f"$qv%.5f", regime) ++ sensVec ++ hVec).mkString(","))
        }
      } finally
        embFile.close()

      // Also export a 2D Policy Response Surface grid for beta = weight
      val surfFile = new java.io.PrintWriter(new java.io.File(outDir, s"surface_weight-$weight.csv"))
      try {
        surfFile.println("selfDeltaMag,maxNeighDeltaMag,childrenRatio,minNeighAction,predDt,qValue")
        val gridVals = (0 to 25).map(_ / 25.0f)
        for {
          selfMag <- gridVals
          neighMag <- gridVals
          childR <- Seq(0.0f, 0.5f)
        } {
          val deltaFlag = if (selfMag > 0.05f) 1.0f else 0.0f
          val neighRatio = if (neighMag > 0.05f) math.min(1.0f, neighMag * 0.8f) else 0.0f
          val minAct = if (neighMag > 0.1f) 0.3f else 0.8f
          val row =
            Array(deltaFlag, 0.0f, selfMag, neighMag, neighRatio, neighMag * 0.5f, childR, minAct, weight.toFloat)
          val tIn = sub.create(row, new Shape(1, stateDim))
          val scOut = actor.evaluate(tIn)
          val qOut = critic1.evaluate(tIn.concat(scOut, 1))
          val dt = actionScaler.unscale(scOut.getFloat(0L))
          val qv = qOut.getFloat(0L)
          surfFile.println(f"$selfMag%.4f,$neighMag%.4f,$childR%.2f,$minAct%.2f,$dt%.5f,$qv%.5f")
        }
      } finally
        surfFile.close()
    }
  }

  private def counterfactualReward(rawIntensity: Double, dt: Double, beta: Double): Float =
    TradeOff.offlineReward(rawIntensity, dt, beta, min, max).toFloat

  private def withWeight(baseFeatures: Array[Float], beta: Double): Array[Float] = {
    val copy = baseFeatures.clone()
    copy(copy.length - 1) = TradeOff.betaFeature(beta)
    copy
  }

  private def appendEpisodeTrajectories(episodes: Seq[Trajectory]): Unit =
    episodes.foreach { trajectory =>
      val baseStates = trajectory.map(step => encode(step.state, weight)).toIndexedSeq
      val intensities = trajectory.map(step => SpatialTemporal.instabilityIntensity(step.state)).toIndexedSeq
      val weightsToTrain = Array(weight, weight) ++ counterfactualWeights
      trajectory.indices.foreach { i =>
        val rawAction = trajectory(i).action
        val isTerminal = i == trajectory.size - 1
        val nextBase = if (isTerminal) baseStates(i) else baseStates(i + 1)
        val scaledAct = actionScaler.scale(rawAction)
        val smdpDiscount = math.pow(gamma.toDouble, math.max(0.5, rawAction)).toFloat
        val intensity = intensities(i)
        val targetBuffer = if (intensity > 0.0) unstableTransitions else stableTransitions
        weightsToTrain.foreach { beta =>
          targetBuffer += OfflineLearnerContinuous.ContinuousTransition(
            state = withWeight(baseStates(i), beta),
            scaledAction = scaledAct,
            reward = counterfactualReward(intensity, rawAction, beta),
            nextState = withWeight(nextBase, beta),
            terminal = if (isTerminal) 1.0f else 0.0f,
            discountFactor = smdpDiscount
          )
        }
      }
    }

  private def sampleStratifiedBatch(effectiveBatch: Int): Vector[OfflineLearnerContinuous.ContinuousTransition] = {
    if (unstableTransitions.isEmpty) {
      Vector.fill(effectiveBatch)(stableTransitions(random.nextInt(stableTransitions.size)))
    } else if (stableTransitions.isEmpty) {
      Vector.fill(effectiveBatch)(unstableTransitions(random.nextInt(unstableTransitions.size)))
    } else {
      val half = effectiveBatch / 2
      val unstPart = Vector.fill(half)(unstableTransitions(random.nextInt(unstableTransitions.size)))
      val stabPart = Vector.fill(effectiveBatch - half)(stableTransitions(random.nextInt(stableTransitions.size)))
      unstPart ++ stabPart
    }
  }

  private def fit(nSteps: Int): Unit = {
    val dataSize = unstableTransitions.size + stableTransitions.size
    if (dataSize == 0) return
    val effectiveBatch = math.min(batchSize, dataSize)

    (0 until nSteps).foreach { _ =>
      val batch = sampleStratifiedBatch(effectiveBatch)
      withSubManager(rootManager) { sub =>
        val states = sub.create(batch.map(_.state).toArray)
        val nextStates = sub.create(batch.map(_.nextState).toArray)
        val actions = sub.create(batch.map(_.scaledAction).toArray, new Shape(effectiveBatch, 1))
        val rewards = sub.create(batch.map(_.reward).toArray, new Shape(effectiveBatch, 1))
        val terminals = sub.create(batch.map(_.terminal).toArray, new Shape(effectiveBatch, 1))
        val discounts = sub.create(batch.map(_.discountFactor).toArray, new Shape(effectiveBatch, 1))

        val nextActions = targetActor.evaluate(nextStates)
        val nextStateActions = nextStates.concat(nextActions, 1)
        val nextQ = targetCritic1.evaluate(nextStateActions).minimum(targetCritic2.evaluate(nextStateActions))
        val targetQ = (rewards + nextQ * discounts * (-terminals + 1.0f)).stopGradient()

        val stateActions = states.concat(actions, 1)
        critic1.trainStep {
          TorchNetwork.smoothL1Loss(critic1.forward(stateActions, training = true), targetQ)
        }
        critic2.trainStep {
          TorchNetwork.smoothL1Loss(critic2.forward(stateActions, training = true), targetQ)
        }
        critic1.withFrozenParameters {
          actor.trainStep {
            val piActions = actor.forward(states, training = true)
            val qValues = critic1.forward(states.concat(piActions, 1), training = true)
            -qValues.mean()
          }
        }

        targetCritic1.softUpdateFrom(critic1, targetUpdateTau)
        targetCritic2.softUpdateFrom(critic2, targetUpdateTau)
        targetActor.softUpdateFrom(actor, targetUpdateTau)
      }
    }
  }
}

object OfflineLearnerContinuous {
  final case class ContinuousTransition(
      state: Array[Float],
      scaledAction: Float,
      reward: Float,
      nextState: Array[Float],
      terminal: Float,
      discountFactor: Float
  )
}
