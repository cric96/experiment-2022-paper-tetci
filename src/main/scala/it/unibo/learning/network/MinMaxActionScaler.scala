package it.unibo.learning.network

/** Maps actions in [`min`, `max`] to the [−1, 1] range of a tanh output layer, and back; both directions clamp. */
final case class MinMaxActionScaler(min: Double, max: Double) {
  require(max > min, s"max ($max) must be greater than min ($min)")
  private val halfRange = (max - min) / 2.0
  private val mid = (max + min) / 2.0

  def scale(action: Double): Float = ((math.max(min, math.min(max, action)) - mid) / halfRange).toFloat

  def unscale(scaledAction: Double): Double = mid + math.max(-1.0, math.min(1.0, scaledAction)) * halfRange
}
