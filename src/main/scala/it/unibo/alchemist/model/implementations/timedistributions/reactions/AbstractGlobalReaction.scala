package it.unibo.alchemist.model.implementations.timedistributions.reactions

import it.unibo.alchemist.model.interfaces._
import org.danilopianini.util.{ListSet, ListSets}

import java.util
import scala.jdk.CollectionConverters._

/** A global reaction scheduled by `distribution`, without conditions nor dependencies, whose behaviour is given by
  * [[executeBeforeUpdateDistribution]].
  */
abstract class AbstractGlobalReaction[T, P <: Position[P]](
    val environment: Environment[T, P],
    distribution: TimeDistribution[T]
) extends GlobalReaction[T] {
  private val actions: util.List[Action[T]] = new util.ArrayList()
  private val conditions: util.List[Condition[T]] = new util.ArrayList()

  /** What the reaction does when it fires. */
  protected def executeBeforeUpdateDistribution(): Unit

  override def execute(): Unit = {
    executeBeforeUpdateDistribution()
    distribution.update(getTimeDistribution.getNextOccurence, true, getRate, environment)
  }

  override def getActions: util.List[Action[T]] = actions

  override def setActions(list: util.List[_ <: Action[T]]): Unit = {
    actions.clear()
    actions.addAll(list)
  }

  override def getConditions: util.List[Condition[T]] = conditions

  override def setConditions(list: util.List[_ <: Condition[T]]): Unit = {
    conditions.clear()
    conditions.addAll(list)
  }

  override def getInboundDependencies: ListSet[_ <: Dependency] = ListSets.emptyListSet()

  override def getOutboundDependencies: ListSet[_ <: Dependency] = ListSets.emptyListSet()

  override def getTimeDistribution: TimeDistribution[T] = distribution

  /** Global reactions have no conditions: they always fire when scheduled. */
  override def canExecute: Boolean = true

  override def initializationComplete(time: Time, environment: Environment[T, _]): Unit = ()

  override def update(time: Time, executed: Boolean, environment: Environment[T, _]): Unit = ()

  override def compareTo(other: Actionable[T]): Int = getTau.compareTo(other.getTau)

  override def getRate: Double = distribution.getRate

  override def getTau: Time = distribution.getNextOccurence

  /** The nodes currently in the environment. */
  def agents: List[Node[T]] = environment.getNodes.iterator().asScala.toList
}
