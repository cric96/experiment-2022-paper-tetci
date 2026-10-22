package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.ReplayBuffer.Experience

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Fixed-capacity ring buffer of experiences shared by all the nodes; once full, the oldest experience is replaced. */
final class ReplayBuffer(capacity: Int, random: Random):
  private val buffer: ArrayBuffer[Experience] = ArrayBuffer.empty
  private var writePosition: Int = 0

  def put(stateT: AgentState, actionT: Int, trueAction: Double, rewardTPlus: Double, stateTPlus: AgentState): Unit =
    if capacity > 0 then
      val experience = Experience(stateT, actionT, trueAction, rewardTPlus, stateTPlus)
      if buffer.size < capacity then buffer += experience
      else
        buffer(writePosition) = experience
        writePosition = (writePosition + 1) % capacity

  /** Samples `batchSize` distinct experiences uniformly (all of them, shuffled, if the buffer is smaller). */
  def sample(batchSize: Int): Seq[Experience] =
    if buffer.isEmpty || batchSize <= 0 then Seq.empty
    else if batchSize >= buffer.size then random.shuffle(buffer.toSeq)
    else Iterator.continually(random.nextInt(buffer.size)).distinct.take(batchSize).map(buffer).toSeq

  def totalSize: Int = buffer.size

  def all: Seq[Experience] = buffer.toSeq

object ReplayBuffer:
  final case class Experience(
      stateT: AgentState,
      actionT: Int,
      trueAction: Double,
      rewardTPlus: Double,
      stateTPlus: AgentState
  )
