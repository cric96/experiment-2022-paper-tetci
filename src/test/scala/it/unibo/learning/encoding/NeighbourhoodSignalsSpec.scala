package it.unibo.learning.encoding

import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NeighbourhoodSignalsSpec extends AnyFlatSpec with Matchers:
  "The Bellman-Ford residual" should "measure how much the node's edges drifted" in {
    NeighbourhoodSignals.bellmanFordResidual(sourceMovingTowardsMe) shouldBe 0.9999999958776927
    fixtures.drop(1).map(NeighbourhoodSignals.bellmanFordResidual) shouldBe Seq(0.0, 0.0, 0.0)
  }

  "The downstream children ratio" should "count the neighbours that route through the node" in {
    fixtures.map(NeighbourhoodSignals.downstreamChildrenRatio) shouldBe Seq(1.0, 0.0, 0.5, 0.0)
  }

  "The local scale" should "be a fifth of the largest finite output, at least 1, 50 when unknown" in {
    fixtures.map(NeighbourhoodSignals.localScale) shouldBe Seq(2.0, 2.0, 4.0, 50.0)
  }

  "The neighbourhood change" should "report the largest normalized change and the share of changed neighbours" in {
    val change = NeighbourhoodSignals.neighbourhoodChange(neighbourBecameUnreachable, scale = 2.0)
    change.maxMagnitude shouldBe math.tanh(NeighbourhoodSignals.UnboundedChangeInScales)
    change.changedRatio shouldBe 0.5
    NeighbourhoodSignals.neighbourhoodChange(isolated, scale = 2.0) shouldBe
      NeighbourhoodSignals.NeighbourhoodChange.None
  }
