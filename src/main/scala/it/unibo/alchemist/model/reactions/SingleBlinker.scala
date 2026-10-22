package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.model.reactions.SingleBlinker.*
import it.unibo.alchemist.model.scafi.nodes.SimpleNodeManager
import it.unibo.alchemist.model.{Environment, Node, Position, TimeDistribution}
import it.unibo.scafi.Sensors
import org.apache.commons.math3.random.{RandomGenerator, Well19937c}

/** Moves the source of the building blocks at every activation: to a random node farther than `minimumDistance` from
  * the current source (to the farthest node if there is none), or to a random node if there is no source yet. Every
  * activation draws from its own seed, so the moves do not depend on how many nodes the previous ones considered.
  */
final class SingleBlinker[T, P <: Position[P]](
    environment: Environment[T, P],
    random: RandomGenerator,
    distribution: TimeDistribution[T],
    val minimumDistance: Double
) extends AbstractGlobalReaction[T, P](environment, distribution) {
  private val initialSeed: Long = random.nextLong()
  private val choices = new Well19937c(initialSeed)
  private var activations: Int = 0

  override protected def react(): Unit = {
    choices.setSeed(initialSeed ^ (activations.toLong * ActivationSeedMultiplier))
    activations += 1
    val sources = agents.filter(_.valueOf[Boolean](Sensors.source).getOrElse(false))
    sources.foreach(setSource(_, isSource = false))
    sources.headOption.fold(randomNode)(farFrom).foreach(setSource(_, isSource = true))
  }

  private def randomNode: Option[Node[T]] = {
    val nodes = agents
    Option.when(nodes.nonEmpty)(nodes(choices.nextInt(nodes.size)))
  }

  private def farFrom(source: Node[T]): Option[Node[T]] = {
    val origin = environment.getCurrentPosition(source)
    def distance(node: Node[T]): Double = environment.getCurrentPosition(node).distanceTo(origin)
    val candidates = agents.filter(distance(_) > minimumDistance)
    if (candidates.nonEmpty) Some(candidates(choices.nextInt(candidates.size)))
    else agents.maxByOption(distance)(Ordering.Double.TotalOrdering)
  }

  private def setSource(node: Node[T], isSource: Boolean): Unit =
    new SimpleNodeManager[T](node).put(Sensors.source, isSource)
}

object SingleBlinker {

  /** Spreads the activation counter before mixing it with the initial seed. */
  val ActivationSeedMultiplier: Long = 987654321L
}
