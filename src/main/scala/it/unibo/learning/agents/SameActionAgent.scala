package it.unibo.learning.agents

import it.unibo.learning.abstractions.AgentState

/** Fixed-period baseline: every node always wakes up after `action` time units. */
final class SameActionAgent(action: Double) extends Learner:
  override def policy: AgentState => Double = _ => action
