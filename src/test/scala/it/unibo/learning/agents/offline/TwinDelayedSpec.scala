package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDManager
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.collection.mutable.ListBuffer
import scala.util.{Random, Using}

class TwinDelayedSpec extends AnyFlatSpec with Matchers:
  private val tolerance = 1e-6

  /** Target actions at the bounds and in the middle of the scaled action range, each repeated. */
  private val targetActions: Array[Float] = Array.fill(100)(Array(-1.0f, 0.0f, 1.0f)).flatten

  private def smoothed(seed: Int): Seq[Float] =
    Using.resource(NDManager.newBaseManager()) { manager =>
      val actions = manager.create(targetActions).reshape(targetActions.length.toLong, 1)
      TwinDelayed.smoothed(actions, new Random(seed)).toFloatArray.toSeq
    }

  "Delayed policy updates" should "run once every PolicyDelay critic updates" in {
    val updates = TwinDelayed.DelayedPolicyUpdates()
    val updatedAt = ListBuffer.empty[Int]
    (1 to 3 * TwinDelayed.PolicyDelay).foreach(criticUpdate => updates.afterCriticUpdate(updatedAt += criticUpdate))
    updatedAt.toSeq shouldBe (1 to 3).map(_ * TwinDelayed.PolicyDelay)
  }

  "Target policy smoothing" should "perturb the target actions by at most the noise clip" in {
    smoothed(seed = 1).zip(targetActions).foreach { case (noisy, action) =>
      math.abs(noisy - action).toDouble should be <= TwinDelayed.NoiseClip + tolerance
    }
    smoothed(seed = 1).zip(targetActions).count { case (noisy, action) => noisy != action } should be > 0
  }

  it should "keep the target actions within the scaled action range" in {
    all(smoothed(seed = 2).map(_.toDouble)) should (be >= -1.0 and be <= 1.0)
  }

  it should "draw the noise from the given source of randomness" in {
    smoothed(seed = 3) shouldBe smoothed(seed = 3)
    smoothed(seed = 3) should not be smoothed(seed = 4)
  }
end TwinDelayedSpec
