package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.model.reactions.SingleBlinker.*
import it.unibo.alchemist.model.scafi.nodes.SimpleNodeManager
import it.unibo.alchemist.model.{Environment, Node, Position, TimeDistribution}
import it.unibo.scafi.{BuildingBlock, Sensors}
import it.unibo.inspection.FailedNodeMetrics
import org.apache.commons.math3.random.{RandomGenerator, Well19937c}

import scala.jdk.CollectionConverters.*

/** Moves the source of the building blocks at every activation: to a random node farther than `minimumDistance` from
  * the current source (to the farthest node if there is none), or to a random node if there is no source yet. Every
  * activation draws from its own seed, so the moves do not depend on how many nodes the previous ones considered.
  * During leader-election evaluation, the mid-episode activation removes an elected leader instead of moving the
  * source.
  */
final class SingleBlinker[T, P <: Position[P]](
    environment: Environment[T, P],
    random: RandomGenerator,
    distribution: TimeDistribution[T],
    val minimumDistance: Double
) extends AbstractGlobalReaction[T, P](environment, distribution):
  private val initialSeed: Long = random.nextLong()
  private val choices = new Well19937c(initialSeed)
  private var activations: Int = 0

  /** Counters retained until the next episode after a node is removed. */
  val failedMetrics: FailedNodeMetrics = FailedNodeMetrics()

  private def electionIsActive: Boolean =
    val target = agents.headOption.flatMap(_.valueOf[Any](Sensors.program)).map(BuildingBlock.fromMolecule)
    environment.getReactions.getCurrent.asScala
      .collectFirst { case central: CentralAgent[?, ?] => central }
      .exists(central => target.exists(block => central.activeBlock(block) == BuildingBlock.LeaderElection))

  override protected def react(): Unit =
    choices.setSeed(initialSeed ^ (activations.toLong * ActivationSeedMultiplier))
    activations += 1
    val sources = agents.filter(_.valueOf[Boolean](Sensors.source).getOrElse(false))
    if electionIsActive && sources.nonEmpty then removeLeader()
    else
      if sources.isEmpty then failedMetrics.reset()
      sources.foreach(setSource(_, isSource = false))
      sources.headOption.fold(randomNode)(farFrom).foreach(setSource(_, isSource = true))

  /** Fail an actually elected leader, preferring the boosted candidate, then the lowest identifier. */
  private def removeLeader(): Unit =
    val leaders = agents.filter(_.valueOf[Boolean](Sensors.electedLeader).getOrElse(false))
    val victim = leaders
      .minByOption(node => (if node.valueOf[Boolean](Sensors.source).getOrElse(false) then 0 else 1, node.getId))
      .getOrElse(throw new IllegalStateException("No elected leader is available at the failure event"))
    val counters = RetainedCounters.map { molecule =>
      molecule -> victim.valueOf[Number](molecule).fold(0.0)(_.doubleValue())
    }.toMap
    failedMetrics.record(counters)
    environment.removeNode(victim)

  private def randomNode: Option[Node[T]] =
    val nodes = agents
    Option.when(nodes.nonEmpty)(nodes(choices.nextInt(nodes.size)))

  private def farFrom(source: Node[T]): Option[Node[T]] =
    val origin = environment.getCurrentPosition(source)
    def distance(node: Node[T]): Double = environment.getCurrentPosition(node).distanceTo(origin)
    val candidates = agents.filter(distance(_) > minimumDistance)
    if candidates.nonEmpty then Some(candidates(choices.nextInt(candidates.size)))
    else agents.maxByOption(distance)(Ordering.Double.TotalOrdering)

  private def setSource(node: Node[T], isSource: Boolean): Unit =
    SimpleNodeManager[T](node).put(Sensors.source, isSource)
end SingleBlinker

object SingleBlinker:

  /** Episode counters that must survive physical node removal. */
  val RetainedCounters: Seq[String] = Seq(
    Sensors.ticks,
    Sensors.accumulatedReward,
    Sensors.accumulatedUnstableTime,
    Sensors.accumulatedError
  )

  /** Spreads the activation counter before mixing it with the initial seed. */
  val ActivationSeedMultiplier: Long = 987654321L
