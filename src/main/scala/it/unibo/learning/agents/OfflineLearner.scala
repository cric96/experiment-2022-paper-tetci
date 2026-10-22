package it.unibo.learning.agents

import ai.djl.ndarray.{NDList, NDManager}
import ai.djl.ndarray.types.Shape
import ai.djl.training.loss.Loss
import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, Step, Trajectory}
import it.unibo.learning.network.NeuralNetworkRL.SpatialTemporal
import it.unibo.learning.network.TorchNetwork
import it.unibo.learning.network.TorchNetwork._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class OfflineLearner(
    actionSpace: Seq[Double],
    randomKey: Double,
    weight: Double,
    val stateDim: Int = SpatialTemporal.FeatureDim,
    val hiddenSize: Int = 64,
    val gamma: Float = 0.9f,
    val bcqThreshold: Float = 0.08f,
    val lr: Float = 1e-3f,
    val batchSize: Int = 64,
    val steps: Int = 160,
    val explorationSteps: Int = 10,
    val learningStepAfter: Int = 3
) extends Learner {

  private val numActions: Int = actionSpace.size
  private val maxAction: Double = actionSpace.max
  private var random: Random = new Random((randomKey.toLong * 31L) ^ java.lang.Double.doubleToLongBits(weight))
  private val rootManager: NDManager = NDManager.newBaseManager()

  private val baseSeed: Int = (randomKey.toInt * 1009) ^ weight.hashCode()
  private val qNet: TorchNetwork =
    TorchNetwork.createMLP("bcq-q", stateDim, Seq(hiddenSize, hiddenSize), numActions, lr = lr, seed = baseSeed + 1)
  private val targetQNet: TorchNetwork =
    TorchNetwork.createMLP(
      "bcq-target-q",
      stateDim,
      Seq(hiddenSize, hiddenSize),
      numActions,
      lr = lr,
      seed = baseSeed + 2
    )
  targetQNet.softUpdateFrom(qNet, 1.0f)

  private val imitatorNet: TorchNetwork =
    TorchNetwork.createMLP(
      "bcq-imitator",
      stateDim,
      Seq(hiddenSize, hiddenSize),
      numActions,
      lr = lr,
      seed = baseSeed + 3
    )
  private val crossEntropyLoss: Loss = Loss.softmaxCrossEntropyLoss()

  private val initialEpsilon: Double = 0.99
  private val finalEpsilon: Double = 0.0
  private val targetUpdateTau: Float = 0.02f
  private val counterfactualWeights: Array[Double] = Array(0.5, 0.8, 0.95, 0.99, 1.0)

  private var ticks: Int = 0
  private val unstableTransitions: ArrayBuffer[OfflineLearner.DiscreteTransition] = ArrayBuffer.empty
  private val stableTransitions: ArrayBuffer[OfflineLearner.DiscreteTransition] = ArrayBuffer.empty
  @volatile private var fastQEval: Array[Float] => Array[Float] = qNet.snapshotFastMLP(useTanh = false)
  @volatile private var fastImEval: Array[Float] => Array[Float] = imitatorNet.snapshotFastMLP(useTanh = false)

  private def encode(state: AgentState, targetWeight: Double = weight): Array[Float] =
    SpatialTemporal.encode(state, targetWeight, maxAction)

  private def currentEpsilon: Double =
    if (ticks >= explorationSteps) finalEpsilon
    else initialEpsilon + (finalEpsilon - initialEpsilon) * (ticks.toDouble / explorationSteps.toDouble)

  private def selectBCQAction(stateFeatures: Array[Float]): Int = {
    val qValues = fastQEval(stateFeatures)
    val logits = fastImEval(stateFeatures)
    var maxLogit = Float.NegativeInfinity
    var i = 0
    while (i < numActions) {
      if (logits(i) > maxLogit) maxLogit = logits(i)
      i += 1
    }
    val probs = new Array[Float](numActions)
    var sumExp = 0.0f
    i = 0
    while (i < numActions) {
      val e = math.exp(logits(i) - maxLogit).toFloat
      probs(i) = e
      sumExp += e
      i += 1
    }
    var maxProb = 0.0f
    i = 0
    while (i < numActions) {
      probs(i) /= sumExp
      if (probs(i) > maxProb) maxProb = probs(i)
      i += 1
    }
    val thresh = bcqThreshold * maxProb
    var bestAct = 0
    var bestQ = Float.NegativeInfinity
    i = 0
    while (i < numActions) {
      if (probs(i) >= thresh && qValues(i) > bestQ) {
        bestQ = qValues(i)
        bestAct = i
      }
      i += 1
    }
    bestAct
  }

  override def policyForWeight(customWeight: Double): AgentState => Double = { state =>
    val actionIdx = if (ticks < explorationSteps && random.nextDouble() < currentEpsilon) {
      random.nextInt(numActions)
    } else {
      selectBCQAction(encode(state, customWeight))
    }
    actionSpace(actionIdx)
  }

  override def policy: AgentState => Double = policyForWeight(weight)

  override def injectRandom(random: Random): Unit = this.random = random

  override def endEpisode(trajectories: Seq[Trajectory]): Unit = {
    appendEpisodeTrajectories(trajectories)
    if (ticks <= explorationSteps && (unstableTransitions.nonEmpty || stableTransitions.nonEmpty)) {
      val nSteps = if (ticks < explorationSteps) steps else steps * learningStepAfter
      fit(nSteps)
      fastQEval = qNet.snapshotFastMLP(useTanh = false)
      fastImEval = imitatorNet.snapshotFastMLP(useTanh = false)
    }
    ticks += 1
  }

  private def counterfactualReward(rawIntensity: Double, dt: Double, beta: Double): Float =
    TradeOff.offlineReward(rawIntensity, dt, beta, TradeOff.FastestPeriod, maxAction).toFloat

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
        val Step(_, actionIdx, rawAction, _) = trajectory(i)
        if (actionIdx >= 0 && actionIdx < numActions) {
          val isTerminal = i == trajectory.size - 1
          val nextBase = if (isTerminal) baseStates(i) else baseStates(i + 1)
          // SMDP continuous-time discount factor: gamma^(Delta t)
          val smdpDiscount = math.pow(gamma.toDouble, math.max(0.5, rawAction)).toFloat
          val intensity = intensities(i)
          val targetBuffer = if (intensity > 0.0) unstableTransitions else stableTransitions
          weightsToTrain.foreach { beta =>
            targetBuffer += OfflineLearner.DiscreteTransition(
              state = withWeight(baseStates(i), beta),
              action = actionIdx,
              reward = counterfactualReward(intensity, rawAction, beta),
              nextState = withWeight(nextBase, beta),
              terminal = if (isTerminal) 1.0f else 0.0f,
              discountFactor = smdpDiscount
            )
          }
        }
      }
    }

  private def sampleStratifiedBatch(effectiveBatch: Int): Vector[OfflineLearner.DiscreteTransition] = {
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
        val actions = sub.create(batch.map(_.action.toLong).toArray, new Shape(effectiveBatch, 1))
        val rewards = sub.create(batch.map(_.reward).toArray, new Shape(effectiveBatch, 1))
        val terminals = sub.create(batch.map(_.terminal).toArray, new Shape(effectiveBatch, 1))
        val discounts = sub.create(batch.map(_.discountFactor).toArray, new Shape(effectiveBatch, 1))

        val nextQ = qNet.evaluate(nextStates)
        val nextProbs = imitatorNet.evaluate(nextStates).softmax(1)
        val maxNextProbs = nextProbs.max(Array(1), true)
        val allowedMask = nextProbs.div(maxNextProbs).gte(bcqThreshold)
        val maskedNextQ = nextQ.duplicate()
        maskedNextQ.set(allowedMask.logicalNot(), -1e8f)
        val bestNextActions = maskedNextQ.argMax(1).reshape(effectiveBatch, 1)
        val nextTargetQ = targetQNet.evaluate(nextStates).gather(bestNextActions, 1)
        val targetValues = (rewards + nextTargetQ * discounts * (-terminals + 1.0f)).stopGradient()

        qNet.trainStep {
          val currentQ = qNet.forward(states, training = true).gather(actions, 1)
          TorchNetwork.smoothL1Loss(currentQ, targetValues)
        }

        imitatorNet.trainStep {
          val logits = imitatorNet.forward(states, training = true)
          val ceLoss = crossEntropyLoss.evaluate(new NDList(actions.squeeze(1)), new NDList(logits))
          val l2Penalty = logits.pow(2).mean() * 1e-2f
          ceLoss + l2Penalty
        }

        targetQNet.softUpdateFrom(qNet, targetUpdateTau)
      }
    }
  }
}

object OfflineLearner {
  final case class DiscreteTransition(
      state: Array[Float],
      action: Int,
      reward: Float,
      nextState: Array[Float],
      terminal: Float,
      discountFactor: Float
  )
}
