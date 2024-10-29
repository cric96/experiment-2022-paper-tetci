package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.{AgentState, ReplayBuffer}

import scala.util.Random

class SameActionAgent(action: Double) extends Learner {

  override def policy: AgentState => Double = _ => action

  override def store(where: String): Unit = {}

  override def load(where: String): AgentState => Double = _ => action

  override def update(batch: Seq[ReplayBuffer.Experience]): Unit = {}

  override def injectRandom(random: Random): Unit = {}

  override def injectCentralAgent(agent: CentralAgent[_, _]): Unit = {}
}
