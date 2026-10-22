package it.unibo.learning.agents

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.network.NeuralNetworkRL.{SpatialTemporal, StarGraph}
import it.unibo.learning.network.TorchNetwork
import it.unibo.learning.network.TorchNetwork._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Edge-Conditioned Graph Attention Network (EdgeGAT) + Vectorial 2-Head Multi-Objective Critic
  * ([Q_conv, Q_energy] in R^2) for Continuous Offline Reinforcement Learning on Star-Graphs.
  */
class OfflineLearnerGNN(
    min: Double,
    max: Double,
    randomKey: Double,
    weight: Double,
    val edgeHidden: Int = 24,
    val readoutHidden: Int = 48,
    val gamma: Float = 0.9f,
    val lr: Float = 1e-3f,
    val batchSize: Int = 64,
    val steps: Int = 110,
    val explorationSteps: Int = 10,
    val learningStepAfter: Int = 2
) extends Learner {

  private val selfDim: Int = StarGraph.SelfDim
  private val maxNeighbors: Int = StarGraph.MaxNeighbors
  private val neighDim: Int = StarGraph.NeighDim
  private val stateDim: Int = StarGraph.TotalDim

  private var random: Random = new Random((randomKey.toLong * 31L) ^ java.lang.Double.doubleToLongBits(weight))
  private val rootManager: NDManager = NDManager.newBaseManager()
  private val actionScaler: TorchNetwork.MinMaxActionScaler = TorchNetwork.MinMaxActionScaler(min, max)
  private val baseSeed: Int = (randomKey.toInt * 1009) ^ weight.hashCode()

  // EdgeGAT Actor: StarGraph(selfDim=5, K=8, neighDim=6) -> action in [-1, 1]
  private val actor: TorchNetwork = TorchNetwork.createEdgeGAT(
    name = "gat-actor",
    selfDim = selfDim,
    maxNeighbors = maxNeighbors,
    neighDim = neighDim,
    edgeHidden = edgeHidden,
    readoutHidden = readoutHidden,
    outputSize = 1,
    outputActivation = Some(Activation.tanhBlock()),
    lr = lr * 0.5f,
    seed = baseSeed + 101
  )
  private val targetActor: TorchNetwork = TorchNetwork.createEdgeGAT(
    name = "gat-target-actor",
    selfDim = selfDim,
    maxNeighbors = maxNeighbors,
    neighDim = neighDim,
    edgeHidden = edgeHidden,
    readoutHidden = readoutHidden,
    outputSize = 1,
    outputActivation = Some(Activation.tanhBlock()),
    lr = lr * 0.5f,
    seed = baseSeed + 102
  )
  targetActor.softUpdateFrom(actor, 1.0f)

  // Vectorial 2-Head Multi-Objective EdgeGAT Critics: predict [Q_conv, Q_energy] in R^2
  private val critic1: TorchNetwork = TorchNetwork.createEdgeGAT(
    name = "gat-critic-1",
    selfDim = selfDim + 1,
    maxNeighbors = maxNeighbors,
    neighDim = neighDim,
    edgeHidden = edgeHidden,
    readoutHidden = readoutHidden,
    outputSize = 2,
    lr = lr,
    seed = baseSeed + 103
  )
  private val targetCritic1: TorchNetwork = TorchNetwork.createEdgeGAT(
    name = "gat-target-critic-1",
    selfDim = selfDim + 1,
    maxNeighbors = maxNeighbors,
    neighDim = neighDim,
    edgeHidden = edgeHidden,
    readoutHidden = readoutHidden,
    outputSize = 2,
    lr = lr,
    seed = baseSeed + 104
  )
  targetCritic1.softUpdateFrom(critic1, 1.0f)

  private val critic2: TorchNetwork = TorchNetwork.createEdgeGAT(
    name = "gat-critic-2",
    selfDim = selfDim + 1,
    maxNeighbors = maxNeighbors,
    neighDim = neighDim,
    edgeHidden = edgeHidden,
    readoutHidden = readoutHidden,
    outputSize = 2,
    lr = lr,
    seed = baseSeed + 105
  )
  private val targetCritic2: TorchNetwork = TorchNetwork.createEdgeGAT(
    name = "gat-target-critic-2",
    selfDim = selfDim + 1,
    maxNeighbors = maxNeighbors,
    neighDim = neighDim,
    edgeHidden = edgeHidden,
    readoutHidden = readoutHidden,
    outputSize = 2,
    lr = lr,
    seed = baseSeed + 106
  )
  targetCritic2.softUpdateFrom(critic2, 1.0f)

  private val epsilon: Double = 0.99
  private val targetUpdateTau: Float = 0.02f
  private val counterfactualWeights: Array[Double] = Array(0.5, 0.8, 0.95, 0.99, 1.0)

  private var ticks: Int = 0
  private val unstableTransitions: ArrayBuffer[OfflineLearnerGNN.GNNTransition] = ArrayBuffer.empty
  private val stableTransitions: ArrayBuffer[OfflineLearnerGNN.GNNTransition] = ArrayBuffer.empty
  @volatile private var fastActorEval: Array[Float] => Float =
    actor.block.asInstanceOf[TorchNetwork.EdgeGATBlock].snapshotFastInference()
  @volatile private var fastEmbAndAttn: Array[Float] => (Array[Float], Array[Float]) =
    actor.block.asInstanceOf[TorchNetwork.EdgeGATBlock].snapshotEmbeddingAndAttention()

  def inspectEmbeddingAndAttention(state: AgentState, targetWeight: Double = weight): (Array[Float], Array[Float], List[Int]) = {
    val feat = encode(state, targetWeight)
    val (emb, alphas) = fastEmbAndAttn(feat)
    val currSnap = state.neighborhoodOutput.lastOption.getOrElse(Map.empty)
    val sortedNeighborIds = currSnap
      .filter(_._1 != state.me)
      .toList
      .sortBy(_._2.distance)
      .take(maxNeighbors)
      .map(_._1)
    (emb, alphas, sortedNeighborIds)
  }

  private def encode(state: AgentState, targetWeight: Double = weight): Array[Float] =
    StarGraph.encode(state, targetWeight, max)

  def linearScaler(start: Double, end: Double, totalSteps: Int, current: Int): Double =
    if (current >= totalSteps) end
    else start + (end - start) * current.toDouble / totalSteps.toDouble

  override def policyForWeight(customWeight: Double): AgentState => Double = { state =>
    val choice = random.nextDouble()
    if (choice < linearScaler(epsilon, 0.0, explorationSteps, ticks)) {
      min + random.nextDouble() * (max - min)
    } else {
      val scaledAction = fastActorEval(encode(state, customWeight)).toDouble
      actionScaler.unscale(scaledAction)
    }
  }

  override def policy: AgentState => Double = policyForWeight(weight)

  override def store(where: String): Unit = {}

  override def load(where: String): AgentState => Double = policy

  override def update(batch: Seq[Experience]): Unit = {}

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: CentralAgent[_, _]): Unit = {}

  override def endEpisode(replayBuffer: Seq[Seq[(AgentState, Int, Double, Double)]]): Unit = {
    appendEpisodeTrajectories(replayBuffer)
    if (ticks <= explorationSteps && (unstableTransitions.nonEmpty || stableTransitions.nonEmpty)) {
      val nSteps = if (ticks < explorationSteps) steps else steps * learningStepAfter
      fit(nSteps)
      val gatBlock = actor.block.asInstanceOf[TorchNetwork.EdgeGATBlock]
      fastActorEval = gatBlock.snapshotFastInference()
      fastEmbAndAttn = gatBlock.snapshotEmbeddingAndAttention()
    }
    ticks += 1
  }

  private def objectiveWeightsAndRewards(rawIntensity: Double, dt: Double, beta: Double): (Float, Float, Float, Float) = {
    val effectiveIntensity = 0.06 + 8.0 * rawIntensity
    val u = math.max(0.0, math.min(1.0, (dt - min) / (max - min)))
    val energyWeight = math.max(0.005, 1.0 - beta)
    val wConv = beta * effectiveIntensity
    val normSum = wConv + energyWeight
    val w1 = (wConv / normSum).toFloat
    val w2 = (energyWeight / normSum).toFloat
    val rConv = (-(u * u)).toFloat
    val rEnergy = (-((1.0 - u) * (1.0 - u))).toFloat
    (w1, w2, rConv, rEnergy)
  }

  private def withWeight(baseFeatures: Array[Float], beta: Double): Array[Float] = {
    val copy = baseFeatures.clone()
    val energyW = math.max(0.005, 1.0 - beta)
    copy(4) = (2.0 * (energyW / (energyW + 0.06 * beta)) - 1.0).toFloat
    copy
  }

  private def appendEpisodeTrajectories(episodes: Seq[Seq[(AgentState, Int, Double, Double)]]): Unit =
    episodes.foreach { trajectory =>
      val baseStates = trajectory.map(step => encode(step._1, weight)).toIndexedSeq
      val intensities = trajectory.map(step => SpatialTemporal.instabilityIntensity(step._1)).toIndexedSeq
      val weightsToTrain = Array(weight, weight) ++ counterfactualWeights
      trajectory.indices.foreach { i =>
        val (_, _, rawAction, _) = trajectory(i)
        val isTerminal = i == trajectory.size - 1
        val nextBase = if (isTerminal) baseStates(i) else baseStates(i + 1)
        val scaledAct = actionScaler.scale(rawAction)
        val smdpDiscount = math.pow(gamma.toDouble, math.max(0.5, rawAction)).toFloat
        val intensity = intensities(i)
        val targetBuffer = if (intensity > 0.0) unstableTransitions else stableTransitions
        weightsToTrain.foreach { beta =>
          val (w1, w2, rConv, rEnergy) = objectiveWeightsAndRewards(intensity, rawAction, beta)
          targetBuffer += OfflineLearnerGNN.GNNTransition(
            state = withWeight(baseStates(i), beta),
            scaledAction = scaledAct,
            rConv = rConv,
            rEnergy = rEnergy,
            wConv = w1,
            wEnergy = w2,
            nextState = withWeight(nextBase, beta),
            terminal = if (isTerminal) 1.0f else 0.0f,
            discountFactor = smdpDiscount
          )
        }
      }
    }

  private def sampleStratifiedBatch(effectiveBatch: Int): Vector[OfflineLearnerGNN.GNNTransition] = {
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

  /** Inserts the action tensor (B, 1) right after selfFeat (B, selfDim) before neighbor features (B, K * neighDim). */
  private def insertActionIntoGraphState(states: ai.djl.ndarray.NDArray, actions: ai.djl.ndarray.NDArray): ai.djl.ndarray.NDArray = {
    val selfPart = states.get(new NDIndex(":, :{}", java.lang.Long.valueOf(selfDim)))
    val neighPart = states.get(new NDIndex(":, {}:", java.lang.Long.valueOf(selfDim)))
    selfPart.concat(actions, 1).concat(neighPart, 1)
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
        val rewardsVec = sub.create(batch.map(t => Array(t.rConv, t.rEnergy)).toArray)
        val weightsVec = sub.create(batch.map(t => Array(t.wConv, t.wEnergy)).toArray)
        val terminals = sub.create(batch.map(_.terminal).toArray, new Shape(effectiveBatch, 1))
        val discounts = sub.create(batch.map(_.discountFactor).toArray, new Shape(effectiveBatch, 1))

        val nextActions = targetActor.evaluate(nextStates)
        val nextGraphSa = insertActionIntoGraphState(nextStates, nextActions)
        val nextQ1 = targetCritic1.evaluate(nextGraphSa)
        val nextQ2 = targetCritic2.evaluate(nextGraphSa)
        val nextQVec = nextQ1.minimum(nextQ2)
        val targetQVec = (rewardsVec + nextQVec * discounts * (-terminals + 1.0f)).stopGradient()

        val graphSa = insertActionIntoGraphState(states, actions)
        critic1.trainStep {
          TorchNetwork.smoothL1Loss(critic1.forward(graphSa, training = true), targetQVec)
        }
        critic2.trainStep {
          TorchNetwork.smoothL1Loss(critic2.forward(graphSa, training = true), targetQVec)
        }
        critic1.withFrozenParameters {
          actor.trainStep {
            val piActions = actor.forward(states, training = true)
            val piGraphSa = insertActionIntoGraphState(states, piActions)
            val qVec = critic1.forward(piGraphSa, training = true)
            // Scalarized multi-objective Q value: w_conv(s, beta) * Q_conv + w_energy(s, beta) * Q_energy
            val scalarizedQ = (qVec * weightsVec).sum(Array(1))
            -scalarizedQ.mean()
          }
        }

        targetCritic1.softUpdateFrom(critic1, targetUpdateTau)
        targetCritic2.softUpdateFrom(critic2, targetUpdateTau)
        targetActor.softUpdateFrom(actor, targetUpdateTau)
      }
    }
  }
}

object OfflineLearnerGNN {
  final case class GNNTransition(
      state: Array[Float],
      scaledAction: Float,
      rConv: Float,
      rEnergy: Float,
      wConv: Float,
      wEnergy: Float,
      nextState: Array[Float],
      terminal: Float,
      discountFactor: Float
  )
}
