package it.unibo.learning.abstractions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DecayReferenceSpec extends AnyFlatSpec with Matchers {
  "A constant reference" should "never change" in {
    val reference = DecayReference.constant(0.3)
    reference.update()
    reference.value shouldBe 0.3
  }

  "An exponential decay" should "multiply its value by (1 - rate) at every update" in {
    val reference = DecayReference.exponentialDecay(initialValue = 0.8, rate = 0.5)
    reference.value shouldBe 0.8
    reference.update()
    reference.update()
    reference.value shouldBe 0.2
  }

  "A bounded decay" should "never go below its minimum" in {
    val reference = DecayReference.exponentialDecay(initialValue = 0.8, rate = 0.5).bounded(0.3)
    reference.update()
    reference.value shouldBe 0.4
    reference.update()
    reference.value shouldBe 0.3
  }
}
