package it.unibo.learning.encoding

import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class InstabilityIntensitySpec extends AnyFlatSpec with Matchers {
  "The instability intensity" should "be maximal when the node's own output just changed" in {
    InstabilityIntensity.of(sourceMovingTowardsMe) shouldBe 1.0
  }

  it should "be partial when only a neighbour changed" in {
    InstabilityIntensity.of(neighbourBecameUnreachable) shouldBe 0.7230206850568627
  }

  it should "be zero in a stable neighbourhood" in {
    InstabilityIntensity.of(stableBackbone) shouldBe 0.0
    InstabilityIntensity.of(isolated) shouldBe 0.0
  }
}
