package it.unibo.alchemist.model.implementations.timedistributions

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.{Environment, Node, Time}
import org.apache.commons.math3.random.RandomGenerator

class NextWakeUp[T](val node: Node[T], val time: Time,  molecule: String, val randomGenerator: RandomGenerator) extends AbstractDistribution[T](time) {

  val manager = new SimpleNodeManager[T](node)
  def this(node: Node[T], molecule: String, randomGenerator: RandomGenerator) {
    this(node, Time.ZERO, molecule, randomGenerator)
  }
  override def updateStatus(currentTime: Time, executed: Boolean, param: Double, environment: Environment[T, _]): Unit = {
    val dt = manager.get[Double](molecule)
    setNextOccurrence(currentTime.plus(new DoubleTime(dt)))
  }

  override def cloneOnNewNode(destination: Node[T], currentTime: Time): AbstractDistribution[T] = new NextWakeUp[T](destination, currentTime, molecule, randomGenerator)

  override def getRate: Double = manager.get[Double](molecule)
}