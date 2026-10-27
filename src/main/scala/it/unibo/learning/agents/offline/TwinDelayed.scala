package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDArray

import scala.util.Random

/** The two mechanisms that TD3 adds to the clipped double-Q targets of twin critics: the actor and the target networks
  * are updated once every [[PolicyDelay]] critic updates, and the target actions are smoothed with clipped Gaussian
  * noise. Actions are scaled to the [−1, 1] range of the actors' tanh output.
  */
object TwinDelayed:

  /** Critic updates per update of the actor and of the target networks. */
  val PolicyDelay: Int = 2

  /** Standard deviation of the Gaussian noise added to the target actions. */
  val TargetNoise: Double = 0.2

  /** Largest absolute value of the noise added to a target action. */
  val NoiseClip: Double = 0.5

  /** Lower bound of the scaled actions. */
  val MinScaledAction: Float = -1.0f

  /** Upper bound of the scaled actions. */
  val MaxScaledAction: Float = 1.0f

  /** Counts the critic updates of a learner, across its training sessions. */
  final class DelayedPolicyUpdates:
    private var criticUpdates: Int = 0

    /** Runs `policyUpdate` if this critic update is a multiple of [[PolicyDelay]]. */
    def afterCriticUpdate(policyUpdate: => Unit): Unit =
      criticUpdates += 1
      if criticUpdates % PolicyDelay == 0 then policyUpdate

  /** The target `actions` plus clipped Gaussian noise drawn from `random`, kept within the scaled action range. */
  def smoothed(actions: NDArray, random: Random): NDArray =
    val noise = Array.fill(Math.toIntExact(actions.size()))(clippedNoise(random).toFloat)
    actions.add(actions.getManager.create(noise, actions.getShape)).clip(MinScaledAction, MaxScaledAction)

  private def clippedNoise(random: Random): Double =
    math.max(-NoiseClip, math.min(NoiseClip, random.nextGaussian() * TargetNoise))
end TwinDelayed
