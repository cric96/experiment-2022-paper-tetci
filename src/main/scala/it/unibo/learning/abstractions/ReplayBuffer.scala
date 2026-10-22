package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.ReplayBuffer.Experience

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class ReplayBuffer(size: Int, random: Random) {
  private val buffer: ArrayBuffer[Experience] = new ArrayBuffer[Experience](math.max(16, math.min(size, 1024)))
  private var writePos: Int = 0

  def put(stateT: AgentState, actionT: Int, trueAction: Double, rewardTplus: Double, stateTPlus: AgentState): Unit = {
    if (size <= 0) return
    val exp = Experience(stateT, actionT, trueAction, rewardTplus, stateTPlus)
    if (buffer.size < size) {
      buffer += exp
    } else {
      buffer(writePos) = exp
      writePos = (writePos + 1) % size
    }
  }

  def sample(batchSize: Int): Seq[Experience] = {
    val n = buffer.size
    if (n == 0 || batchSize <= 0) {
      Seq.empty
    } else if (batchSize >= n) {
      random.shuffle(buffer.toSeq)
    } else {
      val indices = new Array[Int](batchSize)
      val chosen = scala.collection.mutable.BitSet.empty
      var i = 0
      while (i < batchSize) {
        var candidate = random.nextInt(n)
        while (chosen.contains(candidate))
          candidate = random.nextInt(n)
        chosen.add(candidate)
        indices(i) = candidate
        i += 1
      }
      indices.map(buffer).toSeq
    }
  }

  def totalSize: Int = buffer.size

  def all: Seq[Experience] = buffer.toSeq
}

object ReplayBuffer {
  case class Experience(
      stateT: AgentState,
      actionT: Int,
      trueAction: Double,
      rewardTPlus: Double,
      stateTPlus: AgentState
  )
}
