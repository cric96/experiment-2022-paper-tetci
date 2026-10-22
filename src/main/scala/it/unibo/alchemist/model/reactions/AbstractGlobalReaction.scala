package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.model.timedistributions.AbstractDistribution
import it.unibo.alchemist.model.{
  Environment,
  Node,
  Position,
  ReactionHost,
  Time,
  TimeDistributedReaction,
  TimeDistribution
}

import scala.jdk.CollectionConverters.*

/** A reaction of the whole environment, fired by `distribution`, whose behaviour is given by [[react]]. */
abstract class AbstractGlobalReaction[T, P <: Position[P]](
    val environment: Environment[T, P],
    distribution: TimeDistribution[T]
) extends AbstractReaction[T](AbstractGlobalReaction.startOf(distribution))
    with TimeDistributedReaction[T] {

  /** What the reaction does when it fires. */
  protected def react(): Unit

  override protected def performModelMutation(): Unit = react()

  override def getHost: ReactionHost[T] = environment

  override def getTimeDistribution: TimeDistribution[T] = distribution

  override def getMeanDelay: Double = distribution.getMeanDelay

  override def updateSchedulingAfterFiring(currentTime: Time): Unit = scheduleAfter(currentTime)

  override protected def scheduleAfterInvalidation(currentTime: Time): Unit = scheduleAfter(currentTime)

  private def scheduleAfter(currentTime: Time): Unit = {
    val start = AbstractGlobalReaction.startOf(distribution)
    val from = if currentTime.compareTo(start) > 0 then currentTime else start
    setNextOccurrence(from.plus(distribution.sample()))
  }

  /** The nodes currently in the environment. */
  def agents: List[Node[T]] = environment.getNodes.getCurrent.asScala.toList
}

object AbstractGlobalReaction {
  private def startOf(distribution: TimeDistribution[?]): Time = distribution match {
    case defined: AbstractDistribution[?] => defined.getStartTime
    case _ => Time.ZERO
  }
}
