package it.unibo.learning.agents

import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.abstractions.{AgentState, DecayReference, Trajectory}
import it.unibo.learning.agents.QLearning.*
import it.unibo.util.TemporalInfo

import scala.util.Random

/** Tabular Q-Learning over a hand-crafted state (own output trends and whether a neighbour changed), trained online
  * from the shared replay buffer during the first `LearningEpisodes` episodes, with semi-Markov discounting.
  */
final class QLearning(
    actionSpace: List[Double],
    epsilon: DecayReference[Double],
    alpha: DecayReference[Double],
    gamma: Double,
    seed: Int
) extends Learner:
  private var random: Random = new Random(seed)
  private var qTable: Map[(QState, Int), Double] = Map.empty.withDefaultValue(0.0)
  private var episodes: Int = 0

  def encodeState(state: AgentState): QState =
    val ownTrends = TemporalInfo.computeDeltaTrend(state.ownOutputs.takeRight(HistoryLength))
    ownTrends :+ (if anyNeighbourChanged(state) then 1.0 else 0.0)

  private def anyNeighbourChanged(state: AgentState): Boolean =
    state.lastTwoSnapshots.exists { case (previous, current) =>
      state.neighboursIn(current).exists { case (id, now) => previous.get(id).exists(_.data != now.data) }
    }

  /** Whether the Q table still learns and the policy still explores: afterwards the policy is frozen and greedy. */
  private def isLearning: Boolean = episodes < LearningEpisodes

  override def policy: AgentState => Double = state =>
    if isLearning && random.nextDouble() < epsilon.value then actionSpace(random.nextInt(actionSpace.size))
    else actionSpace(greedyAction(encodeState(state)))

  /** The index of an action with the highest Q value, ties broken at random. */
  private def greedyAction(state: QState): Int =
    val values = actionSpace.indices.map(action => action -> qTable((state, action)))
    val best = values.map { case (_, value) => value }.max
    val bestActions = values.collect { case (action, value) if value == best => action }
    bestActions(random.nextInt(bestActions.size))

  override def update(batch: Seq[Experience]): Unit =
    if isLearning then batch.filter(experience => actionSpace.indices.contains(experience.actionT)).foreach(learnFrom)

  private def learnFrom(experience: Experience): Unit =
    val state = encodeState(experience.stateT)
    val nextState = encodeState(experience.stateTPlus)
    val current = qTable((state, experience.actionT))
    val nextBest = actionSpace.indices.map(action => qTable((nextState, action))).max
    // Semi-Markov (continuous-time) discounting: gamma^(Delta t)
    val discount = math.pow(gamma, math.max(MinDiscountPeriod, experience.trueAction))
    val temporalDifference = experience.rewardTPlus + (discount * nextBest - current)
    qTable = qTable.updated((state, experience.actionT), current + temporalDifference * alpha.value)

  override def endEpisode(trajectories: Seq[Trajectory]): Unit = episodes += 1

  override def injectRandom(random: Random): Unit = this.random = random

  override def decayingParameters: Seq[(String, DecayReference[Double])] = Seq("epsilon" -> epsilon, "alpha" -> alpha)
end QLearning

object QLearning:

  /** Own output trends followed by 1 if a neighbour changed in the last round, 0 otherwise. */
  type QState = List[Double]

  /** Number of own outputs inspected to compute the trends. */
  val HistoryLength: Int = 4

  /** The Q table is updated only during the first episodes; afterwards the policy is only exploited. */
  val LearningEpisodes: Int = 10

  /** Wake-up periods shorter than this are discounted as if they lasted this long. */
  val MinDiscountPeriod: Double = 1.0
