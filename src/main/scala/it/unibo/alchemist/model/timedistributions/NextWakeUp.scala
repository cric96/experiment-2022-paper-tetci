package it.unibo.alchemist.model.timedistributions

import it.unibo.alchemist.model.scafi.nodes.SimpleNodeManager
import it.unibo.alchemist.model.timedistributions.NextWakeUp.*
import it.unibo.alchemist.model.times.DoubleTime
import it.unibo.alchemist.model.{Node, Time, TimeDistribution}
import org.apache.commons.math3.random.RandomGenerator

/** Schedules the next round of a node after the wake-up period the node wrote in `molecule`.
  *
  * A node deployed by the simulation file runs its first round at a random time in [0, 1). The clone of a node
  * re-deployed during the simulation waits [[RestartDelay]] plus a phase that depends only on its id, for
  * reproducibility.
  */
class NextWakeUp[T] private (
    node: Node[T],
    start: Time,
    molecule: String,
    randomGenerator: RandomGenerator,
    firstDelay: Option[Double]
) extends AbstractDistribution[T](start) {
  private val manager = SimpleNodeManager[T](node)
  private var pendingFirstDelay: Option[Double] = firstDelay

  def this(node: Node[T], molecule: String, randomGenerator: RandomGenerator) =
    this(node, new DoubleTime(randomGenerator.nextDouble()), molecule, randomGenerator, None)

  override def sample(): Time = {
    val delay = pendingFirstDelay.getOrElse(period)
    pendingFirstDelay = None
    new DoubleTime(delay)
  }

  override def getMeanDelay: Double = period

  override def newInstanceOn(destination: Node[T]): TimeDistribution[T] =
    new NextWakeUp[T](
      destination,
      Time.ZERO,
      molecule,
      randomGenerator,
      Some(RestartDelay + phaseOf(destination.getId))
    )

  private def period: Double = manager.get[Double](molecule)
}

object NextWakeUp {

  /** The clones of the re-deployed nodes start this long after being deployed. */
  val RestartDelay: Double = 1.0

  /** Knuth's multiplicative hashing constant, ⌊2^32 / φ⌋, which spreads consecutive ids over 32 bits. */
  val GoldenRatioMultiplier: Long = 2654435761L

  /** Keeps the low 32 bits of the hash. */
  val UnsignedIntMask: Long = 0xffffffffL

  /** 2^32, which maps the 32-bit hash to [0, 1). */
  val HashRange: Double = 4294967296.0

  /** A deterministic phase in [0, 1) for node `id`, well spread across ids. */
  def phaseOf(id: Int): Double = ((id * GoldenRatioMultiplier) & UnsignedIntMask).toDouble / HashRange
}
