package it.unibo.learning.agents.offline

/** Soft-update rates of the target networks used by the offline learners. */
object TargetNetworks:

  /** Polyak rate applied after every gradient step: target ← rate · online + (1 − rate) · target. */
  val UpdateRate: Float = 0.02f

  /** Rate that copies the online network, used to initialize the targets. */
  val HardCopy: Float = 1.0f
