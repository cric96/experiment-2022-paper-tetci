package it.unibo.learning.agents.offline

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RegimeSpec extends AnyFlatSpec with Matchers {
  private def features(
      latestTrend: Float = 0,
      selfChange: Float = 0,
      neighbourChange: Float = 0,
      childrenRatio: Float = 0
  ): Array[Float] = Array(latestTrend, 0, selfChange, neighbourChange, 0, 0, childrenRatio, 1, 0)

  "A node" should "be in steady state when nothing changes" in {
    Regime.of(features()) shouldBe Regime.SteadyState
  }

  it should "receive an early warning when only its neighbourhood changes" in {
    Regime.of(features(neighbourChange = 0.5f)) shouldBe Regime.NeighbourEarlyWarning
  }

  it should "be hit by the wave as a leaf or as part of the backbone, depending on its children" in {
    Regime.of(features(latestTrend = -1, selfChange = 0.3f)) shouldBe Regime.LeafWave
    Regime.of(features(latestTrend = -1, selfChange = 0.3f, childrenRatio = 0.5f)) shouldBe Regime.BackboneWave
  }

  it should "keep the labels read by the plotting scripts" in {
    Regime.values.toSeq.map(_.label) shouldBe Seq("Backbone_Wave", "Leaf_Wave", "Neighbor_EarlyWarning", "Steady_State")
  }
}
