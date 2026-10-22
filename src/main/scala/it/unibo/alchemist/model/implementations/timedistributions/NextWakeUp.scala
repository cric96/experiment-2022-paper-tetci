package it.unibo.alchemist.model.implementations.timedistributions

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.{Environment, Node, Time}
import org.apache.commons.math3.random.RandomGenerator

class NextWakeUp[T](val node: Node[T], val time: Time, molecule: String, val randomGenerator: RandomGenerator)
    extends AbstractDistribution[T](time) {

  val manager = new SimpleNodeManager[T](node)
  def this(node: Node[T], molecule: String, randomGenerator: RandomGenerator) =
    this(node, new DoubleTime(randomGenerator.nextDouble()), molecule, randomGenerator)

  override def updateStatus(
      currentTime: Time,
      executed: Boolean,
      param: Double,
      environment: Environment[T, _]
  ): Unit = {
    val dt = manager.get[Double](molecule)
    val target = currentTime.plus(new DoubleTime(dt))
    if (executed || target.compareTo(getNextOccurence) < 0) {
      setNextOccurrence(target)
    }
  }

  override def cloneOnNewNode(destination: Node[T], currentTime: Time): AbstractDistribution[T] = {
    val offset = ((destination.getId * 2654435761L) & 0xffffffffL).toDouble / 4294967296.0
    new NextWakeUp[T](
      destination,
      currentTime.plus(new DoubleTime(offset)),
      molecule,
      randomGenerator
    )
  }

  override def getRate: Double = manager.get[Double](molecule)
}
