package it.unibo.learning.encoding

import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OwnTrendsSpec extends AnyFlatSpec with Matchers {
  "The own trends" should "list the most recent trends first, padded with zeros" in {
    OwnTrends.recent(sourceMovingTowardsMe, 2).toSeq shouldBe Seq(-1.0, 0.0)
    OwnTrends.recent(isolated, 2).toSeq shouldBe Seq(0.0, 0.0)
  }
}
