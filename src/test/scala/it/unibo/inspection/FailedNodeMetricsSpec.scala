package it.unibo.inspection

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FailedNodeMetricsSpec extends AnyFlatSpec with Matchers:
  "Failed-node metrics" should "retain work done before a failure and reset between episodes" in {
    val metrics = FailedNodeMetrics()
    metrics.record(Map("ticks" -> 12.0, "accumulatedError" -> 3.5))
    metrics.record(Map("ticks" -> 8.0, "accumulatedError" -> 2.0))
    metrics.totalOf("ticks") shouldBe 20.0
    metrics.totalOf("accumulatedError") shouldBe 5.5
    metrics.totalOf("unknown") shouldBe 0.0
    metrics.failedNodes shouldBe 2
    metrics.reset()
    metrics.totalOf("ticks") shouldBe 0.0
    metrics.failedNodes shouldBe 0
  }
