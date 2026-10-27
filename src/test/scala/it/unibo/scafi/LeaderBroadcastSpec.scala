package it.unibo.scafi

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LeaderBroadcastSpec extends AnyFlatSpec with Matchers:
  private val line = Topology(
    Seq(0, 1, 2, 3, 9),
    Map(0 -> Seq(1 -> 160.0), 1 -> Seq(0 -> 160.0, 2 -> 160.0), 2 -> Seq(1 -> 160.0, 3 -> 160.0), 3 -> Seq(2 -> 160.0))
  )

  "The leader broadcast reference" should "assign the nearest elected leader's id to every node" in {
    LeaderElection.broadcastIds(line, Set.empty) shouldBe Map(0 -> 0.0, 1 -> 0.0, 2 -> 3.0, 3 -> 3.0, 9 -> 9.0)
    LeaderElection.broadcastIds(line, Set(1)) shouldBe Map(0 -> 1.0, 1 -> 1.0, 2 -> 1.0, 3 -> 1.0, 9 -> 9.0)
  }

  it should "use replacement ids after the leader is removed" in {
    val survivors = Topology(
      line.nodes.filterNot(_ == 1),
      line.neighbours.view.filterKeys(_ != 1).mapValues(_.filterNot(_._1 == 1)).toMap
    )
    LeaderElection.broadcastIds(survivors, Set.empty) shouldBe Map(0 -> 0.0, 2 -> 2.0, 3 -> 2.0, 9 -> 9.0)
  }

  "Leader ids" should "have categorical error independent of their numeric separation" in {
    BuildingBlock.LeaderElection.absoluteError(1024.0, 1025.0) shouldBe 1.0
    BuildingBlock.LeaderElection.absoluteError(1024.0, 9000.0) shouldBe 1.0
    BuildingBlock.LeaderElection.absoluteError(1024.0, 1024.0) shouldBe 0.0
  }
