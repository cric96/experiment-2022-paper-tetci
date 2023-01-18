package it.unibo.learning.abstractions

import scala.language.implicitConversions

trait DecayReference[V] {
  def update(): Unit = {}
  def value: V
}

object DecayReference {
  def constant[V](constant: V): DecayReference[V] = new DecayReference[V] {
    override val value: V = constant
  }

  def exponentialDecay(initialValue: Double, rate: Double): DecayReference[Double] = new DecayReference[Double] {
    private var ticks: Int = 0
    override def update(): Unit = ticks += 1
    override def value: Double = initialValue * math.pow(1 - rate, ticks)
  }

  implicit def flat[V](value: DecayReference[V]): V = value.value

  implicit class wrapper(ref: DecayReference[Double]) {
    def bounded(min: Double): DecayReference[Double] = new DecayReference[Double] {
      override def update(): Unit = ref.update()
      override def value: Double = if (ref.value < min) min else ref.value
    }
  }
}
