package it.unibo.learning.network

import it.unibo.learning.network.NeuralNetworkRL.{Historical, SpatialTemporal, StarGraph}
import it.unibo.testing.AgentStates._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NeuralNetworkRLSpec extends AnyFlatSpec with Matchers {

  "The instability intensity" should "be maximal when the node's own output just changed" in {
    SpatialTemporal.instabilityIntensity(sourceMovingTowardsMe) shouldBe 1.0
  }

  it should "be partial when only a neighbour changed" in {
    SpatialTemporal.instabilityIntensity(neighbourBecameUnreachable) shouldBe 0.7230206850568627
  }

  it should "be zero in a stable neighbourhood" in {
    SpatialTemporal.instabilityIntensity(stableBackbone) shouldBe 0.0
    SpatialTemporal.instabilityIntensity(isolated) shouldBe 0.0
  }

  "The Bellman-Ford residual" should "measure how much the node's edges drifted" in {
    SpatialTemporal.bellmanFordResidual(sourceMovingTowardsMe) shouldBe 0.9999999958776927
    fixtures.drop(1).map(SpatialTemporal.bellmanFordResidual) shouldBe Seq(0.0, 0.0, 0.0)
  }

  "The downstream children ratio" should "count the neighbours that route through the node" in {
    fixtures.map(SpatialTemporal.downstreamChildrenRatio) shouldBe Seq(1.0, 0.0, 0.5, 0.0)
  }

  "The local scale" should "be a fifth of the largest finite output, at least 1, 50 when unknown" in {
    fixtures.map(SpatialTemporal.localScale) shouldBe Seq(2.0, 2.0, 4.0, 50.0)
  }

  "The historical encoding" should "list the most recent trends first, padded with zeros" in {
    Historical.encodeHistory(sourceMovingTowardsMe, 2).toSeq shouldBe Seq(-1.0, 0.0)
    Historical.encodeHistory(isolated, 2).toSeq shouldBe Seq(0.0, 0.0)
  }

  "The spatial-temporal encoding" should "keep its features" in {
    val beta = -0.06542056f
    fixtures.map(SpatialTemporal.encode(_, 0.95, 5.0).toSeq) shouldBe Seq(
      Seq(-1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.6f, beta),
      Seq(0.0f, 0.0f, 0.0f, 0.9640276f, 0.5f, 0.0f, 0.0f, 0.2f, beta),
      Seq(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.2f, beta),
      Seq(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, beta)
    )
  }

  "The star-graph encoding" should "keep the ego features and one slot per neighbour, nearest first" in {
    val beta = 0.61290324f
    def padding(slots: Int): Seq[Float] = Seq.fill(slots)(Seq(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f)).flatten
    fixtures.map(StarGraph.encode(_, 0.8, 5.0).toSeq) shouldBe Seq(
      Seq(-1.0f, 0.0f, 1.0f, 1.0f, beta) ++ Seq(0.0f, 0.0f, 1.0f, 0.9999092f, 0.1f, 0.6f) ++ padding(7),
      Seq(0.0f, 0.0f, 0.0f, 0.0f, beta) ++ Seq(0.0f, 0.0f, 0.0f, -0.9999092f, 0.1f, 0.2f) ++
        Seq(-1.0f, 0.99999976f, 0.99999976f, -0.9950548f, 0.1f, 0.8f) ++ padding(6),
      Seq(0.0f, 0.0f, 0.0f, 0.5f, beta) ++ Seq(0.0f, 0.0f, 0.0f, -0.9866143f, 0.1f, 0.2f) ++
        Seq(0.0f, 0.0f, 0.0f, 0.9866143f, 0.1f, 0.2f) ++ padding(6),
      Seq(0.0f, 0.0f, 0.0f, 0.0f, beta) ++ padding(8)
    )
  }

  it should "have the declared size" in {
    fixtures.map(StarGraph.encode(_, 0.8, 5.0).length).distinct shouldBe Seq(StarGraph.TotalDim)
  }
}
