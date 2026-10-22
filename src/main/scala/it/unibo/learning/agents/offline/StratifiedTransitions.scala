package it.unibo.learning.agents.offline

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Transitions collected by an offline learner, split by regime: instability is rare, so batches are balanced to
  * contain as many transitions from unstable states as from stable ones.
  */
final class StratifiedTransitions[T] {
  private val unstable: ArrayBuffer[T] = ArrayBuffer.empty
  private val stable: ArrayBuffer[T] = ArrayBuffer.empty

  def add(transition: T, isUnstable: Boolean): Unit =
    if isUnstable then unstable += transition else stable += transition

  def size: Int = unstable.size + stable.size

  def isEmpty: Boolean = size == 0

  /** Samples with replacement half of the batch from the unstable regime and the rest from the stable one (or the whole
    * batch from the only non-empty regime).
    */
  def sample(batchSize: Int, random: Random): Vector[T] =
    if unstable.isEmpty then draw(stable, batchSize, random)
    else if stable.isEmpty then draw(unstable, batchSize, random)
    else {
      val unstableHalf = draw(unstable, batchSize / 2, random)
      unstableHalf ++ draw(stable, batchSize - batchSize / 2, random)
    }

  /** The first `perRegime` transitions of each regime, unstable ones first. */
  def oldest(perRegime: Int): Vector[T] = (unstable.take(perRegime) ++ stable.take(perRegime)).toVector

  private def draw(regime: ArrayBuffer[T], count: Int, random: Random): Vector[T] =
    Vector.fill(count)(regime(random.nextInt(regime.size)))
}
