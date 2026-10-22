package it.unibo.learning.abstractions

/** A hyper-parameter that may decay at the end of every episode (e.g. the exploration rate). */
trait DecayReference[V] {
  def value: V

  /** Advances the decay by one episode. */
  def update(): Unit = ()
}

object DecayReference {
  def constant[V](constant: V): DecayReference[V] = new DecayReference[V] {
    override val value: V = constant
  }

  /** initialValue · (1 − rate)^episodes. */
  def exponentialDecay(initialValue: Double, rate: Double): DecayReference[Double] = new DecayReference[Double] {
    private var episodes: Int = 0
    override def update(): Unit = episodes += 1
    override def value: Double = initialValue * math.pow(1 - rate, episodes)
  }

  extension (reference: DecayReference[Double]) {

    /** The same decay, never below `min`. */
    def bounded(min: Double): DecayReference[Double] = new DecayReference[Double] {
      override def update(): Unit = reference.update()
      override def value: Double = math.max(min, reference.value)
    }
  }
}
