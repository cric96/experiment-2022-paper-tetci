package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.ReplayBuffer.Experience

import scala.util.Random


class ReplayBuffer(size: Int, random: Random) {
  private var queue = List.empty[Experience]
  def put(stateT: AgentState, actionT: Int, rewardTplus: Double, stateTPlus: AgentState): Unit = {
    queue = (Experience(stateT, actionT, rewardTplus, stateTPlus) :: queue).take(size)
  }
  def sample(batchSize: Int): Seq[Experience] = {
    random.shuffle(queue).take(batchSize)
  }

  def totaleSize: Int = queue.size
}
object ReplayBuffer {
  case class Experience(stateT: AgentState, actionT: Int, rewardTPlus: Double, stateTPlus: AgentState)
}
