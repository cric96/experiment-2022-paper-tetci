package it.unibo.learning.agents.offline

import it.unibo.learning.network.NeuralNetworkRL.SpatialTemporal
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PolicySurfaceSpec extends AnyFlatSpec with Matchers {
  private val gridSize = PolicySurface.Resolution + 1

  "The policy surface" should "cover the grid of own and neighbour changes for every children ratio" in {
    PolicySurface.points should have size (gridSize * gridSize * PolicySurface.ChildrenRatios.size).toLong
    PolicySurface.points.map(_.selfChange).distinct should have size gridSize.toLong
    PolicySurface.points.map(_.selfChange).max shouldBe 1.0f
  }

  "A surface point" should "derive plausible features from the change magnitudes" in {
    val quiet = PolicySurface.SurfacePoint(selfChange = 0, neighbourChange = 0, childrenRatio = 0)
    val busy = PolicySurface.SurfacePoint(selfChange = 0.5f, neighbourChange = 0.5f, childrenRatio = 0.5f)
    quiet.features(betaFeature = 0.25f).toSeq shouldBe Seq(0, 0, 0, 0, 0, 0, 0, 0.8f, 0.25f)
    busy.features(betaFeature = 0.25f).toSeq shouldBe Seq(1, 0, 0.5f, 0.5f, 0.4f, 0.25f, 0.5f, 0.3f, 0.25f)
  }

  it should "have as many features as the spatial-temporal encoder" in {
    PolicySurface.points.head.features(0).length shouldBe SpatialTemporal.FeatureDim
  }
}
