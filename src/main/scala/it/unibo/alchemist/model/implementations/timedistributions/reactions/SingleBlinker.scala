package it.unibo.alchemist.model.implementations.timedistributions.reactions

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.{Environment, Position, TimeDistribution}
import org.apache.commons.math3.random.RandomGenerator

class SingleBlinker[T, P <: Position[P]](
  environment: Environment[T, P],
  random: RandomGenerator,
  distribution: TimeDistribution[T],
  val minimumDistance: Double) extends AbstractGlobalReaction[T, P](environment, distribution){

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val source = managers.filter(_.get[Boolean]("source"))
    if(source.nonEmpty) {
      val oldSource = source.head
      oldSource.put("source", false)
      val oldPosition = environment.getPosition(oldSource.node)
      val newSource = agents.find(node => environment.getPosition(node).distanceTo(oldPosition) > minimumDistance)
      newSource.foreach(node => new SimpleNodeManager(node).put("source", true))
    } else {
      val index = random.nextInt(managers.size)
      managers(index).put("source", true)
    }
  }
}
