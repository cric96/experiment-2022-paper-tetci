package it.unibo.learning.agents
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.{AgentState, Contextual, DecayReference, ReplayBuffer}
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

  implicit def stateToEncoding: AgentState => QState = state => {
    val me = state.neighborhoodOutput.map(neigh => neigh(state.me))
    TemporalInfo.computeDeltaTrend(me.map(_.data))
  }

  private var random = new Random()
  private var Q: Map[(QState, Int), Double] = Map.empty.withDefault(_ => 0.0) // all near to 0

  override def policy: AgentState => (Int, Contextual) = if (random.nextDouble() < epsilon.value) { _ =>
    (random.shuffle(actionSpace.indices.toList).head, Contextual.empty)
  } else { state =>
    (actionSpace.indices.map(action => action -> Q((state, action))).maxBy(_._2)._1, Contextual.empty)
  }

  override def store(where: String): Unit = {}

  override def load(where: String): (AgentState, (Int, Contextual)) = null // todo

  override def update(batch: Seq[ReplayBuffer.Experience]): Unit = {
    batch.foreach { experience =>
      val currentValue = Q((experience.stateT, experience.actionT))
      val reward = experience.rewardTPlus
      val nextBest = actionSpace.indices.map(action => Q((experience.stateTPlus, action))).max
      val diff = gamma * nextBest - currentValue
      val delta = reward + diff
      Q = Q.updated((experience.stateT, experience.actionT), currentValue + delta * alpha.value)
    }
  }

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: CentralAgent[_, _]): Unit =
    agent.attachDecayable("epsilon" -> epsilon, "alpha" -> alpha)
}
