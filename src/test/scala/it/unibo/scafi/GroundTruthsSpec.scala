package it.unibo.scafi

import it.unibo.scafi.BuildingBlock.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GroundTruthsSpec extends AnyFlatSpec with Matchers {
  private val pair = Topology(nodes = Seq(0, 1, 5), neighbours = Map(0 -> Seq(1 -> 3.0), 1 -> Seq(0 -> 3.0)))

  "The ground truths" should "give the value of every building block in every node" in {
    val truths = GroundTruths.of(pair, sources = Set(1))
    BuildingBlock.values.toSeq.map(truths.of(0, _)) shouldBe Seq(3.0, 1.0, BuildingBlock.broadcastPayloadOf(1), 1.0)
    BuildingBlock.values.toSeq.map(truths.of(1, _)) shouldBe Seq(0.0, 0.0, BuildingBlock.broadcastPayloadOf(1), 2.0)
  }

  it should "use the values of unreachable nodes for nodes without a path to a source" in {
    val truths = GroundTruths.of(pair, sources = Set(1))
    BuildingBlock.values.toSeq.map(truths.of(5, _)) shouldBe
      Seq(Double.PositiveInfinity, Double.PositiveInfinity, BuildingBlock.broadcastPayloadOf(1), 1.0)
  }

  it should "broadcast nothing without sources" in {
    GroundTruths.of(pair, sources = Set.empty).of(0, Broadcast) shouldBe 0.0
  }
}
