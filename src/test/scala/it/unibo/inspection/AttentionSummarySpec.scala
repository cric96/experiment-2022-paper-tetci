package it.unibo.inspection

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AttentionSummarySpec extends AnyFlatSpec with Matchers:
  private val origin = (0.0, 0.0)

  "The attention summary" should "find the most attended neighbour and the attention-weighted direction" in {
    val summary = AttentionSummary.of(origin, Seq(0.75f, 0.25f), Seq(Some((10.0, 0.0)), Some((0.0, 4.0))))
    summary.maxAttention shouldBe 0.75f
    summary.focus shouldBe ((10.0, 0.0))
    summary.direction shouldBe ((7.5, 1.0))
    summary.entropy shouldBe -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) +- 1e-6
  }

  it should "count unknown neighbours only in the entropy, and ignore negligible weights" in {
    val summary = AttentionSummary.of(origin, Seq(0.5f, 0.5f, 0.0f), Seq(None, Some((2.0, 2.0)), Some((9.0, 9.0))))
    summary.focus shouldBe ((2.0, 2.0))
    summary.direction shouldBe ((1.0, 1.0))
    summary.entropy shouldBe math.log(2.0) +- 1e-6
  }

  it should "focus on the node itself when it has no neighbours" in {
    AttentionSummary.of((3.0, 4.0), Seq.empty, Seq.empty) shouldBe AttentionSummary(0.0f, 0.0, (0.0, 0.0), (3.0, 4.0))
  }
