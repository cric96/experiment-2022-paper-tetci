package it.unibo.learning.agents

import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.nn.SequentialBlock
import ai.djl.training.loss.Loss
import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, Trajectory}
import it.unibo.learning.agents.OfflineLearner._
import it.unibo.learning.agents.offline.BatchTensors.{column, indexColumn, rows}
import it.unibo.learning.agents.offline._
import it.unibo.learning.network.NeuralNetworkRL.SpatialTemporal
import it.unibo.learning.network.{FastInference, TorchNetwork}
import it.unibo.learning.network.NDArrayOps._

/** Discrete Batch-Constrained Q-Learning (BCQ) over the spatial-temporal state, conditioned on the trade-off weight β:
  * a Q network chooses among the actions that an imitation network deems likely enough under the logged behaviour.
  */
final class OfflineLearner(
    actionSpace: Seq[Double],
    randomKey: Double,
    weight: Double,
    stateDim: Int = SpatialTemporal.FeatureDim,
    hiddenSize: Int = 64,
    gamma: Float = 0.9f,
    bcqThreshold: Float = 0.08f,
    lr: Float = 1e-3f,
    batchSize: Int = 64,
    stepsPerEpisode: Int = 160,
    explorationEpisodes: Int = 10,
    finalTrainingMultiplier: Int = 3
) extends EpisodicOfflineLearner[DiscreteTransition](
      randomKey,
      weight,
      TrainingSchedule(explorationEpisodes, stepsPerEpisode, finalTrainingMultiplier),
      batchSize
    ) {
  private val maxAction: Double = actionSpace.max
  private val networkSeed: Int = LearnerSeeds.networks(randomKey, weight)
  private val qNetwork: TorchNetwork[SequentialBlock] = network("bcq-q", seedOffset = 1)
  private val targetQNetwork: TorchNetwork[SequentialBlock] = network("bcq-target-q", seedOffset = 2)
  targetQNetwork.softUpdateFrom(qNetwork, TargetNetworks.HardCopy)
  private val imitationNetwork: TorchNetwork[SequentialBlock] = network("bcq-imitator", seedOffset = 3)
  private val crossEntropyLoss: Loss = Loss.softmaxCrossEntropyLoss()
  @volatile private var fastQValues: Array[Float] => Array[Float] = qNetwork.snapshotFastMLP()
  @volatile private var fastImitationLogits: Array[Float] => Array[Float] = imitationNetwork.snapshotFastMLP()

  private def network(name: String, seedOffset: Int): TorchNetwork[SequentialBlock] =
    TorchNetwork.createMLP(
      name,
      stateDim,
      Seq(hiddenSize, hiddenSize),
      actionSpace.size,
      lr = lr,
      seed = networkSeed + seedOffset
    )

  override def policy: AgentState => Double = policyForWeight(weight)

  override def policyForWeight(customWeight: Double): AgentState => Double = { state =>
    val isRandom = schedule.isExploring(currentEpisode) && random.nextDouble() < schedule.epsilonAt(currentEpisode)
    val action =
      if (isRandom) random.nextInt(actionSpace.size)
      else bcqAction(SpatialTemporal.encode(state, customWeight, maxAction))
    actionSpace(action)
  }

  /** The action with the highest Q value among those whose imitation probability is at least `bcqThreshold` times the
    * probability of the most likely one.
    */
  private def bcqAction(features: Array[Float]): Int = {
    val probabilities = FastInference.softmax(fastImitationLogits(features))
    val threshold = bcqThreshold * probabilities.foldLeft(0.0f)(math.max)
    val qValues = fastQValues(features)
    qValues.indices.filter(action => probabilities(action) >= threshold).maxBy(qValues)(Ordering.Float.IeeeOrdering)
  }

  override protected def record(trajectory: Trajectory): Unit =
    Relabelling
      .relabel(trajectory, weight, gamma.toDouble, encodeForOwnWeight, SpatialTemporal.Feature.Beta)
      .filter(relabelled => actionSpace.indices.contains(relabelled.step.actionIndex))
      .foreach(relabelled => transitions.add(toTransition(relabelled), relabelled.isUnstable))

  private def encodeForOwnWeight(state: AgentState): Array[Float] = SpatialTemporal.encode(state, weight, maxAction)

  private def toTransition(relabelled: RelabelledStep): DiscreteTransition = {
    val step = relabelled.step
    val reward =
      TradeOff.offlineReward(relabelled.intensity, step.action, relabelled.beta, TradeOff.FastestPeriod, maxAction)
    DiscreteTransition(
      state = relabelled.state,
      action = step.actionIndex,
      reward = reward.toFloat,
      nextState = relabelled.nextState,
      terminal = relabelled.terminalFlag,
      discountFactor = relabelled.discount
    )
  }

  override protected def trainOn(batch: Vector[DiscreteTransition], manager: NDManager): Unit = {
    val tensors = DiscreteBatch(batch, manager)
    val targets = bcqTargets(tensors)
    qNetwork.trainStep {
      TorchNetwork.smoothL1Loss(qNetwork.forward(tensors.states, training = true).gather(tensors.actions, 1), targets)
    }
    imitationNetwork.trainStep(imitationLoss(tensors))
    targetQNetwork.softUpdateFrom(qNetwork, TargetNetworks.UpdateRate)
  }

  /** r + γ^Δt · Q_target(s', a*), with a* the best next action among those the imitation network allows. */
  private def bcqTargets(tensors: DiscreteBatch): NDArray = {
    val nextQValues = qNetwork.evaluate(tensors.nextStates)
    val nextProbabilities = imitationNetwork.evaluate(tensors.nextStates).softmax(1)
    val isAllowed = nextProbabilities.div(nextProbabilities.max(Array(1), true)).gte(bcqThreshold)
    val maskedNextQValues = nextQValues.duplicate()
    maskedNextQValues.set(isAllowed.logicalNot(), ForbiddenActionValue)
    val bestNextActions = maskedNextQValues.argMax(1).reshape(tensors.size.toLong, 1)
    val nextValues = targetQNetwork.evaluate(tensors.nextStates).gather(bestNextActions, 1)
    (tensors.rewards + nextValues * tensors.discounts * (-tensors.terminals + 1.0f)).stopGradient()
  }

  /** Cross-entropy with the logged actions, plus an L2 penalty that keeps the logits calibrated. */
  private def imitationLoss(tensors: DiscreteBatch): NDArray = {
    val logits = imitationNetwork.forward(tensors.states, training = true)
    val crossEntropy = crossEntropyLoss.evaluate(new NDList(tensors.actions.squeeze(1)), new NDList(logits))
    crossEntropy + logits.pow(2).mean() * LogitsPenalty
  }

  override protected def onTrained(): Unit = {
    fastQValues = qNetwork.snapshotFastMLP()
    fastImitationLogits = imitationNetwork.snapshotFastMLP()
  }
}

object OfflineLearner {

  /** Q value given to the actions the imitation network forbids, so that they are never the best ones. */
  val ForbiddenActionValue: Float = -1e8f

  /** Weight of the L2 penalty on the imitation logits. */
  val LogitsPenalty: Float = 1e-2f

  final case class DiscreteTransition(
      state: Array[Float],
      action: Int,
      reward: Float,
      nextState: Array[Float],
      terminal: Float,
      discountFactor: Float
  )

  /** A batch of discrete transitions as tensors; the scalar fields are columns. */
  final private case class DiscreteBatch(
      size: Int,
      states: NDArray,
      nextStates: NDArray,
      actions: NDArray,
      rewards: NDArray,
      terminals: NDArray,
      discounts: NDArray
  )

  private object DiscreteBatch {
    def apply(batch: Vector[DiscreteTransition], manager: NDManager): DiscreteBatch =
      DiscreteBatch(
        size = batch.size,
        states = rows(batch.map(_.state), manager),
        nextStates = rows(batch.map(_.nextState), manager),
        actions = indexColumn(batch.map(_.action), manager),
        rewards = column(batch.map(_.reward), manager),
        terminals = column(batch.map(_.terminal), manager),
        discounts = column(batch.map(_.discountFactor), manager)
      )
  }
}
