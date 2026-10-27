package it.unibo

import it.unibo.testing.Simulations
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LeaderFailureSpec extends AnyFlatSpec with Matchers:
  "The election perturbation" should "remove one device only during evaluation and restore it next episode" in {
    val checkpoints = Seq(1999.5, 2099.5, 2100.1, 2199.5, 2202.0)
    val frames = Simulations.run(
      "simulationGeneralization",
      Map("random" -> 2.0, "algoId" -> 8, "progId" -> 4),
      checkpoints
    )
    frames.map(_.nodes) shouldBe Seq(100, 100, 99, 99, 100)
    frames(1).outputSum should be > 0.0
  }
