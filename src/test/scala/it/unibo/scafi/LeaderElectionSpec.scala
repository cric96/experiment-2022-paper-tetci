package it.unibo.scafi

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LeaderElectionSpec extends AnyFlatSpec with Matchers:
  private val line = Topology(
    nodes = Seq(0, 1, 2, 3, 9),
    neighbours = Map(
      0 -> Seq(1 -> (LeaderElection.Grain * 0.4)),
      1 -> Seq(0 -> (LeaderElection.Grain * 0.4), 2 -> (LeaderElection.Grain * 0.4)),
      2 -> Seq(1 -> (LeaderElection.Grain * 0.4), 3 -> (LeaderElection.Grain * 0.4)),
      3 -> Seq(2 -> (LeaderElection.Grain * 0.4))
    )
  )

  "Sparse leader election" should "elect separated candidates and isolated devices without a preferred node" in {
    LeaderElection.reference(line, Set.empty) shouldBe Map(0 -> 1.0, 1 -> 0.0, 2 -> 0.0, 3 -> 1.0, 9 -> 1.0)
  }

  it should "respond to a moved priority boost without making every other candidate ineligible" in {
    LeaderElection.reference(line, Set(1)) shouldBe Map(0 -> 0.0, 1 -> 1.0, 2 -> 0.0, 3 -> 0.0, 9 -> 1.0)
    LeaderElection.reference(line, Set(3)) shouldBe Map(0 -> 1.0, 1 -> 0.0, 2 -> 0.0, 3 -> 1.0, 9 -> 1.0)
  }

  it should "discard claims that exceed the grain and prefer shorter paths on priority ties" in {
    val near = LeaderElection.Claim(0, preferred = true, distance = 20.0)
    val far = near.copy(distance = 80.0)
    LeaderElection.choose(2, preferred = false, Seq(near -> 10.0, far -> 10.0)) shouldBe near.copy(distance = 30.0)
    LeaderElection.choose(2, preferred = false, Seq(far -> LeaderElection.Grain)) shouldBe LeaderElection.ownClaim(
      2,
      false
    )
  }

  it should "heal stale claims after a priority change under different fair round orders" in {
    val initial = line.nodes.map(node => node -> LeaderElection.ownClaim(node, node == 1)).toMap
    def settle(claims: Map[Int, LeaderElection.Claim], preferred: Set[Int], order: Seq[Int]) =
      (0 until 30).foldLeft(claims) { (current, _) =>
        order.foldLeft(current) { (state, node) =>
          val messages = line.neighboursOf(node).map { case (neighbour, distance) => state(neighbour) -> distance }
          state.updated(node, LeaderElection.choose(node, preferred.contains(node), messages))
        }
      }
    Seq(line.nodes, line.nodes.reverse).foreach { order =>
      val before = settle(initial, Set(1), order)
      val after = settle(before, Set(3), order)
      after.map { case (node, claim) => node -> (if claim.isLeader(node) then 1.0 else 0.0) } shouldBe
        LeaderElection.reference(line, Set(3))
    }
  }
  it should "discard a failed leader's stale claims and elect replacements on the surviving graph" in {
    val failed = 1
    val initial = line.nodes.map(node => node -> LeaderElection.ownClaim(failed, true)).toMap
    val survivors = Topology(
      line.nodes.filterNot(_ == failed),
      line.neighbours.view.filterKeys(_ != failed).mapValues(_.filterNot(_._1 == failed)).toMap
    )
    val settled = (0 until 100).foldLeft(initial - failed) { (claims, _) =>
      survivors.nodes.foldLeft(claims) { (state, node) =>
        val messages = survivors.neighboursOf(node).map { case (neighbour, distance) => state(neighbour) -> distance }
        state.updated(node, LeaderElection.choose(node, preferred = false, messages))
      }
    }
    settled.map { case (node, claim) => node -> (if claim.isLeader(node) then 1.0 else 0.0) } shouldBe
      LeaderElection.reference(survivors, Set.empty)
    settled.values.exists(_.candidate == failed) shouldBe false
  }
end LeaderElectionSpec
