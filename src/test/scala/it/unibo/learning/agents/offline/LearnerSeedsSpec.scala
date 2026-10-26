package it.unibo.learning.agents.offline

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LearnerSeedsSpec extends AnyFlatSpec with Matchers:
  "The seeds" should "be the same for the same run and weight" in {
    LearnerSeeds.random(2.0, 0.95).nextLong() shouldBe LearnerSeeds.random(2.0, 0.95).nextLong()
    LearnerSeeds.networks(2.0, 0.95) shouldBe LearnerSeeds.networks(2.0, 0.95)
  }

  it should "differ across runs and across weights" in {
    LearnerSeeds.networks(2.0, 0.95) should not be LearnerSeeds.networks(3.0, 0.95)
    LearnerSeeds.networks(2.0, 0.95) should not be LearnerSeeds.networks(2.0, 0.8)
    LearnerSeeds.random(2.0, 0.95).nextLong() should not be LearnerSeeds.random(2.0, 0.8).nextLong()
  }

  it should "keep the values used by the published experiments" in {
    LearnerSeeds.networks(2.0, 0.95) shouldBe (2 * 1009) ^ 0.95.hashCode()
  }
