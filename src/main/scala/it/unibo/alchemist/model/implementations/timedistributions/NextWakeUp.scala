package it.unibo.alchemist.model.implementations.timedistributions

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.implementations.timedistributions.NextWakeUp._
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.{Environment, Node, Time}
import org.apache.commons.math3.random.RandomGenerator

/** Schedules the next round of a node after the wake-up period the node wrote in `molecule`. */
class NextWakeUp[T](node: Node[T], start: Time, molecule: String, randomGenerator: RandomGenerator)
    extends AbstractDistribution[T](start) {
  private val manager = new SimpleNodeManager[T](node)

  /** The first round happens at a random time in [0, 1). */
  def this(node: Node[T], molecule: String, randomGenerator: RandomGenerator) =
    this(node, new DoubleTime(randomGenerator.nextDouble()), molecule, randomGenerator)

  override def updateStatus(
      currentTime: Time,
      executed: Boolean,
      param: Double,
      environment: Environment[T, _]
  ): Unit = {
    val target = currentTime.plus(new DoubleTime(manager.get[Double](molecule)))
    if (executed || target.compareTo(getNextOccurence) < 0) setNextOccurrence(target)
  }

  /** The clone of a re-deployed node starts after a phase that depends only on its id, for reproducibility. */
  override def cloneOnNewNode(destination: Node[T], currentTime: Time): AbstractDistribution[T] =
    new NextWakeUp[T](
      destination,
      currentTime.plus(new DoubleTime(phaseOf(destination.getId))),
      molecule,
      randomGenerator
    )

  override def getRate: Double = manager.get[Double](molecule)
}

object NextWakeUp {

  /** Knuth's multiplicative hashing constant, ⌊2^32 / φ⌋, which spreads consecutive ids over 32 bits. */
  val GoldenRatioMultiplier: Long = 2654435761L

  /** Keeps the low 32 bits of the hash. */
  val UnsignedIntMask: Long = 0xffffffffL

  /** 2^32, which maps the 32-bit hash to [0, 1). */
  val HashRange: Double = 4294967296.0

  /** A deterministic phase in [0, 1) for node `id`, well spread across ids. */
  def phaseOf(id: Int): Double = ((id * GoldenRatioMultiplier) & UnsignedIntMask).toDouble / HashRange
}
