package it.unibo.learning.abstractions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.collection.immutable.Queue
import scala.util.Random

class ReplayBufferSpec extends AnyFlatSpec with Matchers {
  private val state = AgentState(0, Queue.empty)

  /** A buffer filled with experiences whose action index is 0 until `count`, in insertion order. */
  private def bufferWith(count: Int, capacity: Int, seed: Int = 42): ReplayBuffer = {
    val buffer = new ReplayBuffer(capacity, new Random(seed))
    (0 until count).foreach(i => buffer.put(state, i, i.toDouble, 0.0, state))
    buffer
  }

  private def actionsOf(experiences: Seq[ReplayBuffer.Experience]): Seq[Int] = experiences.map(_.actionT)

  "A ReplayBuffer" should "store experiences up to its capacity" in {
    bufferWith(count = 3, capacity = 5).all.map(_.actionT) shouldBe Seq(0, 1, 2)
  }

  it should "overwrite the oldest experiences once full" in {
    actionsOf(bufferWith(count = 7, capacity = 5).all) shouldBe Seq(5, 6, 2, 3, 4)
  }

  it should "ignore experiences when its capacity is not positive" in {
    bufferWith(count = 3, capacity = 0).totalSize shouldBe 0
  }

  it should "sample nothing when empty or when asked for no experience" in {
    bufferWith(count = 0, capacity = 5).sample(3) shouldBe empty
    bufferWith(count = 3, capacity = 5).sample(0) shouldBe empty
  }

  it should "return a permutation of everything when the batch is larger than the buffer" in {
    actionsOf(bufferWith(count = 4, capacity = 5).sample(10)).sorted shouldBe Seq(0, 1, 2, 3)
  }

  it should "sample distinct experiences" in {
    val sampled = actionsOf(bufferWith(count = 50, capacity = 50).sample(20))
    sampled should have size 20
    sampled.distinct should have size 20
  }

  it should "keep the seeded sampling order (reproducibility of the experiments)" in {
    actionsOf(bufferWith(count = 10, capacity = 10).sample(6)) shouldBe Seq(0, 3, 8, 4, 5, 9)
    actionsOf(bufferWith(count = 4, capacity = 10).sample(4)) shouldBe Seq(3, 1, 0, 2)
  }
}
