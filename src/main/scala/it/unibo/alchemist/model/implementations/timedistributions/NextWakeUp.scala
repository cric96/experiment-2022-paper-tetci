package it.unibo.alchemist.model.implementations.timedistributions

import it.unibo.alchemist.core.interfaces.Scheduler
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.{Actionable, Environment, Node, Time}
import org.apache.commons.math3.random.RandomGenerator

import scala.jdk.CollectionConverters.IteratorHasAsScala

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

  /** Event-triggered wake-on-wave: clamps a sleeping neighbor's next wake-up time when an urgent wave arrives. */
  def clampNextWakeUp(environment: Environment[T, _], currentTime: Time, urgentDelay: Double): Unit = {
    val target = currentTime.plus(new DoubleTime(urgentDelay))
    if (target.compareTo(getNextOccurence) < 0 && getNextOccurence.compareTo(currentTime) > 0) {
      val prevNext = getNextOccurence.minus(currentTime).toDouble
      if (prevNext > urgentDelay + 0.25) {
        setNextOccurrence(target)
        manager.put(molecule, urgentDelay)
        NextWakeUp.notifyScheduler(environment, node, this)
      }
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

object NextWakeUp {
  def notifyScheduler[T](environment: Environment[T, _], node: Node[T], dist: NextWakeUp[T]): Unit = {
    val sim = environment.getSimulation
    if (sim != null) {
      node.getReactions.iterator().asScala.find(_.getTimeDistribution eq dist).foreach { reaction =>
        try {
          val field = sim.getClass.getDeclaredField("scheduler")
          field.setAccessible(true)
          val scheduler = field.get(sim).asInstanceOf[Scheduler[T]]
          scheduler.updateReaction(reaction.asInstanceOf[Actionable[T]])
        } catch {
          case _: Throwable =>
        }
      }
    }
  }
}
