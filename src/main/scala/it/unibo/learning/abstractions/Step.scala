package it.unibo.learning.abstractions

/** One round of a node: the observed state, the index of the chosen action in the discrete action space (−1 when the
  * action is continuous), the action itself (the wake-up period) and the reward that followed.
  */
final case class Step(state: AgentState, actionIndex: Int, action: Double, reward: Double)
