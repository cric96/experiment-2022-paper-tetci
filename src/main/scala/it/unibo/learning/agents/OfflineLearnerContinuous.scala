package it.unibo.learning.agents

import ai.djl.ndarray.{NDArray, NDManager}
import ai.djl.nn.{Activation, SequentialBlock}
import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, Trajectory}
import it.unibo.learning.agents.OfflineLearnerContinuous._
import it.unibo.learning.agents.offline.BatchTensors.{column, rows}
import it.unibo.learning.agents.offline._
import it.unibo.learning.network.NeuralNetworkRL.SpatialTemporal
import it.unibo.learning.network.{MinMaxActionScaler, TorchNetwork}
import it.unibo.learning.network.NDArrayOps._

import java.io.File

/** Offline actor-critic with twin critics (clipped double Q-learning) over the spatial-temporal state, conditioned on
  * the trade-off weight β: the actor outputs a continuous wake-up period in [`min`, `max`].
  */
final class OfflineLearnerContinuous(
    min: Double,
    max: Double,
    randomKey: Double,
    weight: Double,
    stateDim: Int = SpatialTemporal.FeatureDim,
    hiddenSize: Int = 64,
    gamma: Float = 0.9f,
    lr: Float = 1e-3f,
    batchSize: Int = 64,
    stepsPerEpisode: Int = 160,
    explorationEpisodes: Int = 10,
    finalTrainingMultiplier: Int = 3,
    inspectionDirectory: File = ActorInspection.DefaultDirectory
) extends EpisodicOfflineLearner[ContinuousTransition](
      randomKey,
      weight,
      TrainingSchedule(explorationEpisodes, stepsPerEpisode, finalTrainingMultiplier),
      batchSize
    ) {
  private val actionScaler: MinMaxActionScaler = MinMaxActionScaler(min, max)
  private val networkSeed: Int = LearnerSeeds.networks(randomKey, weight)
  private val actor: TorchNetwork[SequentialBlock] = actorNetwork("actor", seedOffset = 1)
  private val targetActor: TorchNetwork[SequentialBlock] = actorNetwork("target-actor", seedOffset = 2)
  targetActor.softUpdateFrom(actor, TargetNetworks.HardCopy)
  private val critic1: TorchNetwork[SequentialBlock] = criticNetwork("critic-1", seedOffset = 4)
  private val targetCritic1: TorchNetwork[SequentialBlock] = criticNetwork("target-critic-1", seedOffset = 5)
  targetCritic1.softUpdateFrom(critic1, TargetNetworks.HardCopy)
  private val critic2: TorchNetwork[SequentialBlock] = criticNetwork("critic-2", seedOffset = 6)
  private val targetCritic2: TorchNetwork[SequentialBlock] = criticNetwork("target-critic-2", seedOffset = 7)
  targetCritic2.softUpdateFrom(critic2, TargetNetworks.HardCopy)
  @volatile private var fastActor: Array[Float] => Array[Float] = actor.snapshotFastMLP(useTanh = true)

  private def actorNetwork(name: String, seedOffset: Int): TorchNetwork[SequentialBlock] =
    TorchNetwork.createMLP(
      name,
      stateDim,
      Seq(hiddenSize, hiddenSize),
      outputSize = 1,
      Some(Activation.tanhBlock()),
      lr = lr * ActorLearningRateRatio,
      seed = networkSeed + seedOffset
    )

  /** A critic takes the state followed by the scaled action. */
  private def criticNetwork(name: String, seedOffset: Int): TorchNetwork[SequentialBlock] =
    TorchNetwork.createMLP(name, stateDim + 1, Seq(hiddenSize, hiddenSize), 1, lr = lr, seed = networkSeed + seedOffset)

  override def policy: AgentState => Double = policyForWeight(weight)

  override def policyForWeight(customWeight: Double): AgentState => Double = { state =>
    if (random.nextDouble() < schedule.epsilonAt(currentEpisode)) {
      min + random.nextDouble() * (max - min)
    } else {
      actionScaler.unscale(fastActor(SpatialTemporal.encode(state, customWeight, max))(0).toDouble)
    }
  }

  override protected def record(trajectory: Trajectory): Unit =
    Relabelling
      .relabel(trajectory, weight, gamma.toDouble, encodeForOwnWeight, SpatialTemporal.Feature.Beta)
      .foreach(relabelled => transitions.add(toTransition(relabelled), relabelled.isUnstable))

  private def encodeForOwnWeight(state: AgentState): Array[Float] = SpatialTemporal.encode(state, weight, max)

  private def toTransition(relabelled: RelabelledStep): ContinuousTransition = {
    val period = relabelled.step.action
    ContinuousTransition(
      state = relabelled.state,
      scaledAction = actionScaler.scale(period),
      reward = TradeOff.offlineReward(relabelled.intensity, period, relabelled.beta, min, max).toFloat,
      nextState = relabelled.nextState,
      terminal = relabelled.terminalFlag,
      discountFactor = relabelled.discount
    )
  }

  override protected def trainOn(batch: Vector[ContinuousTransition], manager: NDManager): Unit = {
    val tensors = ContinuousBatch(batch, manager)
    val targets = clippedDoubleQTargets(tensors)
    val stateActions = tensors.states.concat(tensors.actions, 1)
    Seq(critic1, critic2).foreach { critic =>
      critic.trainStep(TorchNetwork.smoothL1Loss(critic.forward(stateActions, training = true), targets))
    }
    critic1.withFrozenParameters(actor.trainStep(actorLoss(tensors.states)))
    targetCritic1.softUpdateFrom(critic1, TargetNetworks.UpdateRate)
    targetCritic2.softUpdateFrom(critic2, TargetNetworks.UpdateRate)
    targetActor.softUpdateFrom(actor, TargetNetworks.UpdateRate)
  }

  /** r + γ^Δt · min(Q1_target, Q2_target)(s', π_target(s')). */
  private def clippedDoubleQTargets(tensors: ContinuousBatch): NDArray = {
    val nextStateActions = tensors.nextStates.concat(targetActor.evaluate(tensors.nextStates), 1)
    val nextValues = targetCritic1.evaluate(nextStateActions).minimum(targetCritic2.evaluate(nextStateActions))
    (tensors.rewards + nextValues * tensors.discounts * (-tensors.terminals + 1.0f)).stopGradient()
  }

  /** The actor maximizes the first critic's value of its own actions. */
  private def actorLoss(states: NDArray): NDArray = {
    val actions = actor.forward(states, training = true)
    -critic1.forward(states.concat(actions, 1), training = true).mean()
  }

  override protected def onTrained(): Unit = {
    fastActor = actor.snapshotFastMLP(useTanh = true)
    if (currentEpisode == explorationEpisodes && math.abs(randomKey - InspectedRun) < RunTolerance) {
      new ActorInspection(actor, critic1, actionScaler, stateDim, inspectionDirectory)
        .export(transitions.oldest(ActorInspection.SamplesPerRegime).map(_.state), weight)
    }
  }
}

object OfflineLearnerContinuous {

  /** The actor learns more slowly than the critics. */
  val ActorLearningRateRatio: Float = 0.5f

  /** Seed of the run whose final policy is inspected and exported. */
  val InspectedRun: Double = 2.0
  val RunTolerance: Double = 1e-6

  final case class ContinuousTransition(
      state: Array[Float],
      scaledAction: Float,
      reward: Float,
      nextState: Array[Float],
      terminal: Float,
      discountFactor: Float
  )

  /** A batch of continuous transitions as tensors; the scalar fields are columns. */
  final private case class ContinuousBatch(
      states: NDArray,
      nextStates: NDArray,
      actions: NDArray,
      rewards: NDArray,
      terminals: NDArray,
      discounts: NDArray
  )

  private object ContinuousBatch {
    def apply(batch: Vector[ContinuousTransition], manager: NDManager): ContinuousBatch =
      ContinuousBatch(
        states = rows(batch.map(_.state), manager),
        nextStates = rows(batch.map(_.nextState), manager),
        actions = column(batch.map(_.scaledAction), manager),
        rewards = column(batch.map(_.reward), manager),
        terminals = column(batch.map(_.terminal), manager),
        discounts = column(batch.map(_.discountFactor), manager)
      )
  }
}
