package it.unibo.scafi

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GroundTruthSpec extends AnyFlatSpec with Matchers {

  /** 0 - 1 - 2 - 3 with 10 m links, plus an isolated node 9. */
  private val line = Topology(
    nodes = Seq(0, 1, 2, 3, 9),
    neighbours =
      Map(0 -> Seq(1 -> 10.0), 1 -> Seq(0 -> 10.0, 2 -> 10.0), 2 -> Seq(1 -> 10.0, 3 -> 10.0), 3 -> Seq(2 -> 10.0))
  )

  /** 0 reaches 3 through 1 (5 + 5) or through 2 (2 + 8): equal lengths, so 3 has two candidate parents. */
  private val diamond = Topology(
    nodes = Seq(0, 1, 2, 3),
    neighbours = Map(
      0 -> Seq(1 -> 5.0, 2 -> 2.0),
      1 -> Seq(0 -> 5.0, 3 -> 5.0),
      2 -> Seq(0 -> 2.0, 3 -> 8.0),
      3 -> Seq(1 -> 5.0, 2 -> 8.0)
    )
  )

  "The shortest paths" should "be the distances from the nearest source, leaving unreachable nodes out" in {
    GroundTruth.shortestPaths(line, Set(0)) shouldBe Map(0 -> 0.0, 1 -> 10.0, 2 -> 20.0, 3 -> 30.0)
    GroundTruth.shortestPaths(line, Set(0, 3)) shouldBe Map(0 -> 0.0, 1 -> 10.0, 2 -> 10.0, 3 -> 0.0)
    GroundTruth.shortestPaths(diamond, Set(0)) shouldBe Map(0 -> 0.0, 1 -> 5.0, 2 -> 2.0, 3 -> 10.0)
  }

  it should "be empty without sources" in {
    GroundTruth.shortestPaths(line, Set.empty) shouldBe empty
  }

  "The hop distances" should "count the links from the nearest source" in {
    GroundTruth.hopDistances(line, Set(1)) shouldBe Map(0 -> 1.0, 1 -> 0.0, 2 -> 1.0, 3 -> 2.0)
    GroundTruth.hopDistances(diamond, Set(0)) shouldBe Map(0 -> 0.0, 1 -> 1.0, 2 -> 1.0, 3 -> 2.0)
  }

  "The collect counts" should "count the nodes of the subtree rooted in each node" in {
    val potentials = GroundTruth.shortestPaths(line, Set(0))
    GroundTruth.collectCounts(line, potentials) shouldBe Map(0 -> 4.0, 1 -> 3.0, 2 -> 2.0, 3 -> 1.0, 9 -> 1.0)
  }

  it should "attach a node to the neighbour with the lowest potential, then the lowest id" in {
    val potentials = Map(0 -> 0.0, 1 -> 5.0, 2 -> 5.0, 3 -> 10.0)
    GroundTruth.collectCounts(diamond, potentials) shouldBe Map(0 -> 4.0, 1 -> 2.0, 2 -> 1.0, 3 -> 1.0)
  }

  it should "be empty without potentials" in {
    GroundTruth.collectCounts(line, Map.empty) shouldBe empty
  }
}
