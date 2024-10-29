package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.abstractions.{AgentState, Contextual, ReplayBuffer}

import scala.util.Random

trait Learner {
  def policy: (AgentState => Double) // current policy
  def store(where: String): Unit // store optimal policy
  def load(where: String): (AgentState => Double) // load policy stored
  def update(batch: Seq[Experience]): Unit // update the internal state following the experience computed
  def injectRandom(random: Random): Unit
  def injectCentralAgent(agent: CentralAgent[_, _]): Unit

  def endEpisode(replyBuffer: Seq[Seq[(AgentState, Int, Double, Double)]]): Unit = {}
}
