package it.unibo.learning.agents

import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SameActionAgentSpec extends AnyFlatSpec with Matchers:
  "A SameActionAgent" should "always choose its action, whatever the state and the trade-off" in {
    val agent = SameActionAgent(3.0)
    fixtures.map(agent.policy) shouldBe Seq.fill(fixtures.size)(3.0)
    fixtures.map(agent.policyForWeight(0.5)) shouldBe Seq.fill(fixtures.size)(3.0)
  }
