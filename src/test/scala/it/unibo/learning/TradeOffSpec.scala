package it.unibo.learning

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TradeOffSpec extends AnyFlatSpec with Matchers {
  private val tolerance = 1e-9

  "The energy weight" should "be 1 - beta, but never below its minimum" in {
    TradeOff.energyWeight(0.8) shouldBe 0.2 +- tolerance
    TradeOff.energyWeight(1.0) shouldBe TradeOff.MinEnergyWeight
  }

  "The beta feature" should "map the steady-state energy share to [-1, 1]" in {
    TradeOff.betaFeature(0.5).toDouble shouldBe (2 * 0.5 / (0.5 + 0.06 * 0.5) - 1) +- 1e-6
    TradeOff.betaFeature(1.0).toDouble shouldBe (2 * 0.005 / (0.005 + 0.06) - 1) +- 1e-6
  }

  it should "decrease as convergence becomes more important" in {
    val features = Seq(0.5, 0.8, 0.95, 0.99, 1.0).map(TradeOff.betaFeature)
    features shouldBe features.sorted.reverse
  }

  "The normalized period" should "map [min, max] to [0, 1] and clamp outside" in {
    TradeOff.normalizedPeriod(0.5, min = 0.5, max = 5.0) shouldBe 0.0
    TradeOff.normalizedPeriod(2.75, min = 0.5, max = 5.0) shouldBe 0.5 +- tolerance
    TradeOff.normalizedPeriod(9.0, min = 0.5, max = 5.0) shouldBe 1.0
    TradeOff.normalizedPeriod(0.1, min = 0.5, max = 5.0) shouldBe 0.0
  }

  "The objective weights" should "be normalized" in {
    val weights = TradeOff.objectiveWeights(intensity = 0.3, beta = 0.95)
    weights.convergence + weights.energy shouldBe 1.0 +- tolerance
  }

  it should "favour convergence when the node is unstable" in {
    val stable = TradeOff.objectiveWeights(intensity = 0.0, beta = 0.95)
    val unstable = TradeOff.objectiveWeights(intensity = 1.0, beta = 0.95)
    unstable.convergence should be > stable.convergence
  }

  "The offline reward" should "penalize slow rounds by the convergence weight and fast ones by the energy weight" in {
    val weights = TradeOff.objectiveWeights(intensity = 0.4, beta = 0.8)
    TradeOff.offlineReward(intensity = 0.4, period = 5.0, beta = 0.8, min = 0.5, max = 5.0) shouldBe
      -weights.convergence +- tolerance
    TradeOff.offlineReward(intensity = 0.4, period = 0.5, beta = 0.8, min = 0.5, max = 5.0) shouldBe
      -weights.energy +- tolerance
  }

  "The online reward" should "weight the convergence penalty by beta and the instability intensity" in {
    val u = TradeOff.normalizedPeriod(2.0, min = 0.5, max = 5.0)
    val expected = -0.9 * (0.06 + 0.94 * 0.5) * u * u - 0.1 * (1 - u) * (1 - u)
    TradeOff.onlineReward(intensity = 0.5, period = 2.0, beta = 0.9, min = 0.5, max = 5.0) shouldBe
      expected +- tolerance
  }
}
