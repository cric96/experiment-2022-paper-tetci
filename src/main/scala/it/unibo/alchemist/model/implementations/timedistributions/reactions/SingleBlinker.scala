package it.unibo.alchemist.model.implementations.timedistributions.reactions

import it.unibo.alchemist.EnvironmentOps
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.{Environment, Node, Position, TimeDistribution}
import it.unibo.scafi.Sensors
import org.apache.commons.math3.random.RandomGenerator

import scala.jdk.CollectionConverters.CollectionHasAsScala

class SingleBlinker[T, P <: Position[P]](
    environment: Environment[T, P],
    random: RandomGenerator,
    distribution: TimeDistribution[T],
    val minimumDistance: Double
) extends AbstractGlobalReaction[T, P](environment, distribution) {

  private val initialSeed: Long = random.nextLong()
  private val blinkerRng = new org.apache.commons.math3.random.Well19937c(initialSeed)
  private var blinkStep = 0

  override protected def executeBeforeUpdateDistribution(): Unit = {
    blinkerRng.setSeed(initialSeed ^ (blinkStep.toLong * 987654321L))
    blinkStep += 1
    val source = managers.filter(_.get[Boolean]("source"))
    source.foreach(_.put("source", false))
    if (source.nonEmpty) {
      val oldSource = source.head
      val oldPosition = environment.getPosition(oldSource.node)
      val candidates = agents.filter(node => environment.getPosition(node).distanceTo(oldPosition) > minimumDistance)
      val newSource = if (candidates.nonEmpty) {
        Some(candidates(blinkerRng.nextInt(candidates.size)))
      } else {
        agents.maxByOption(node => environment.getPosition(node).distanceTo(oldPosition))
      }
      newSource.foreach { node =>
        val reference = new SimpleNodeManager(node)
        reference.put("source", true)
      }
    } else if (managers.nonEmpty) {
      val index = blinkerRng.nextInt(managers.size)
      managers(index).put("source", true)
    }
  }

}
