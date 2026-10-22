package it.unibo.learning.agents

import it.unibo.testing.AgentStates
import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HeuristicAdaptiveAgentSpec extends AnyFlatSpec with Matchers {
  private val agent = new HeuristicAdaptiveAgent(Seq(1.0, 2.0, 3.0, 4.0, 5.0), weight = 0.95)
  private val changedTwoRoundsAgo = AgentStates.of(
    me = 0,
    Map(0 -> Reading(20.0)),
    Map(0 -> Reading(0.0)),
    Map(0 -> Reading(0.0))
  )

  "The heuristic" should "wake up fast right after its own output changed" in {
    agent.policy(sourceMovingTowardsMe) shouldBe 1.125
  }

  it should "ramp up towards the steady period after a change" in {
    agent.policy(changedTwoRoundsAgo) shouldBe 2.196787383177571
  }

  it should "sleep at the steady period when its own output is stable" in {
    Seq(neighbourBecameUnreachable, stableBackbone, isolated).map(agent.policy) shouldBe
      Seq.fill(3)(2.869158878504674)
  }

  it should "sleep longer when energy matters more" in {
    agent.policyForWeight(0.8)(stableBackbone) shouldBe 4.225806451612904
    agent.policyForWeight(0.8)(sourceMovingTowardsMe) shouldBe 1.5
  }
}
