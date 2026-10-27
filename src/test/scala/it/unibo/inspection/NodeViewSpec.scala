package it.unibo.inspection

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NodeViewSpec extends AnyFlatSpec with Matchers:
  "A node view" should "replace non-finite outputs and ground truths before measuring the error" in {
    val view = NodeView(3, 1.25, -5.5, isSource = true, Double.PositiveInfinity, 12.25, period = 1.5, ticks = 7)
    view.boundedOutput shouldBe NodeView.UnboundedOutput
    view.boundedGroundTruth shouldBe 12.25
    view.error shouldBe NodeView.UnboundedOutput - 12.25
  }

  it should "format its common columns" in {
    NodeView(3, 1.25, -5.5, isSource = true, 12.25, Double.PositiveInfinity, period = 1.5, ticks = 7).columns shouldBe
      Seq("3", "1.25", "-5.50", "1", "12.25", "600.00", "1.5000", "587.75", "0")
  }

  it should "export the actual leader flag independently of the broadcast id and use categorical error" in {
    val follower = NodeView(3, 0.0, 0.0, false, 10.0, 99.0, 1.0, 1, isLeader = false, categoricalOutput = true)
    follower.error shouldBe 1.0
    follower.columns.last shouldBe "0"
    follower.copy(output = 99.0).error shouldBe 0.0
    follower.copy(isLeader = true).columns.last shouldBe "1"
  }
