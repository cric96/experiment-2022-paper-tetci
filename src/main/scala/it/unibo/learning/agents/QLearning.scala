package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.{AgentState, DecayReference, ReplayBuffer}
import it.unibo.util.TemporalInfo

import scala.util.Random

class QLearning(
    actionSpace: List[Double],
    epsilon: DecayReference[Double],
    alpha: DecayReference[Double],
    gamma: Double,
    seed: Int,
    weightConvergence: Double
) extends Learner {
  type QState = List[Double]
  var ticks: Int = 0
  val learningSteps: Int = 10

  def encodeState(state: AgentState): QState = {
    val me = state.neighborhoodOutput.flatMap(_.get(state.me)).takeRight(4)
    val selfTrend = TemporalInfo.computeDeltaTrend(me.map(_.data))
    val snaps = state.neighborhoodOutput.takeRight(2)
    val neighChanged = if (snaps.size == 2) {
      val prev = snaps.head
      val curr = snaps.last
      if (curr.exists { case (id, info) => id != state.me && prev.get(id).exists(p => p.data != info.data) }) 1.0
      else 0.0
    } else 0.0
    selfTrend :+ neighChanged
  }

  private var random = new Random(seed)
  private var Q: Map[(QState, Int), Double] = Map.empty.withDefaultValue(0.0)

  override def policy: AgentState => Double = { state =>
    if (random.nextDouble() < epsilon.value) {
      actionSpace(random.nextInt(actionSpace.size))
    } else {
      val qState = encodeState(state)
      val qValues = actionSpace.indices.map(a => a -> Q((qState, a)))
      val maxQ = qValues.map(_._2).max
      val bestActions = qValues.filter(_._2 == maxQ).map(_._1)
      val chosenAction = bestActions(random.nextInt(bestActions.size))
      actionSpace(chosenAction)
    }
  }

  override def store(where: String): Unit = {}

  override def load(where: String): AgentState => Double = policy

  override def update(batch: Seq[ReplayBuffer.Experience]): Unit = {
    if (ticks > learningSteps) return
    batch.foreach { experience =>
      if (experience.actionT >= 0 && experience.actionT < actionSpace.size) {
        val stateKey = encodeState(experience.stateT)
        val nextStateKey = encodeState(experience.stateTPlus)
        val currentValue = Q((stateKey, experience.actionT))
        val reward = experience.rewardTPlus
        val nextBest = actionSpace.indices.map(action => Q((nextStateKey, action))).max
        // SMDP continuous-time discounting: gamma^(Delta t)
        val smdpDiscount = math.pow(gamma, math.max(1.0, experience.trueAction))
        val diff = smdpDiscount * nextBest - currentValue
        val delta = reward + diff
        Q = Q.updated((stateKey, experience.actionT), currentValue + delta * alpha.value)
      }
    }
  }

  override def endEpisode(replyBuffer: Seq[Seq[(AgentState, Int, Double, Double)]]): Unit = ticks += 1

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: CentralAgent[_, _]): Unit =
    agent.attachDecayable("epsilon" -> epsilon, "alpha" -> alpha)
}
