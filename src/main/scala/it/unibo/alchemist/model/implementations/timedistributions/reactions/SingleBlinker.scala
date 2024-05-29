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

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val source = managers.filter(_.get[Boolean]("source"))
    source.foreach(_.put("source", false))
    if (source.nonEmpty) {
      val oldSource = source.head
      val oldPosition = environment.getPosition(oldSource.node)
      val newSource = agents.find(node => environment.getPosition(node).distanceTo(oldPosition) > minimumDistance)
      newSource.foreach { node =>
        val reference = new SimpleNodeManager(node)
        reference.put("source", true)
        val other = environment.getClonedOfThis(node.getId)
        other.put("source", true)
      }
    } else {
      val index = random.nextInt(managers.size)
      managers(index).put("source", true)
      environment.getClonedOfThis(managers(index).node.getId).put("source", true)
    }
  }

}
