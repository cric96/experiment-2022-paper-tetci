package it.unibo.alchemist.model.timedistributions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NextWakeUpSpec extends AnyFlatSpec with Matchers {
  "The initial phase of a re-deployed node" should "be a deterministic value in [0, 1) spread across ids" in {
    val phases = (0 until 100).map(NextWakeUp.phaseOf)
    phases.foreach(phase => phase should (be >= 0.0 and be < 1.0))
    phases.distinct should have size 100
    NextWakeUp.phaseOf(42) shouldBe NextWakeUp.phaseOf(42)
  }

  it should "keep the values used by the published experiments" in {
    (0 until 3).map(NextWakeUp.phaseOf) shouldBe
      (0 until 3).map(id => ((id * 2654435761L) & 0xffffffffL).toDouble / 4294967296.0)
  }
}
