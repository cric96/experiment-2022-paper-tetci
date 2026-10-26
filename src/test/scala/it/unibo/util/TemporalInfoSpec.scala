package it.unibo.util

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TemporalInfoSpec extends AnyFlatSpec with Matchers:
  "computeDeltaTrend" should "return the sign of each consecutive difference" in {
    TemporalInfo.computeDeltaTrend(List(1.0, 2.0, 2.0, 1.0)) shouldBe List(1.0, 0.0, -1.0)
  }

  it should "return a single stable trend for a single value" in {
    TemporalInfo.computeDeltaTrend(List(4.0)) shouldBe List(0.0)
  }

  it should "return no trend for an empty history" in {
    TemporalInfo.computeDeltaTrend(List.empty) shouldBe empty
  }

  it should "treat non-finite values as -1" in {
    TemporalInfo.computeDeltaTrend(List(Double.PositiveInfinity, 3.0, Double.NaN)) shouldBe List(1.0, -1.0)
  }
