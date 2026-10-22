package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.AgentState.{NeighborInfo, Snapshot}

import scala.collection.immutable.Queue

/** What node `me` observed in its last rounds, oldest snapshot first; every snapshot includes the node itself. */
final case class AgentState(
    me: Int,
    neighborhoodOutput: Queue[Snapshot],
    neighborhoodSensing: Queue[Snapshot]
) {

  /** The node's own outputs, oldest first. */
  def ownOutputs: Queue[Double] = neighborhoodOutput.flatMap(_.get(me)).map(_.data)

  def latestSnapshot: Option[Snapshot] = neighborhoodOutput.lastOption

  /** The two most recent snapshots as (previous, current), if at least two rounds have been observed. */
  def lastTwoSnapshots: Option[(Snapshot, Snapshot)] = neighborhoodOutput.takeRight(2) match {
    case Queue(previous, current) => Some((previous, current))
    case _ => None
  }

  /** The entries of `snapshot` that refer to the neighbours, excluding the node itself. */
  def neighboursIn(snapshot: Snapshot): Snapshot = snapshot.filter { case (id, _) => id != me }
}

object AgentState {

  /** The information of each device of a neighbourhood (node itself included) in one round, by device id. */
  type Snapshot = Map[Int, NeighborInfo]

  /** Output of a device, its distance from the observer and the wake-up period it chose in its last round. */
  final case class NeighborInfo(data: Double, distance: Double, oldAction: Double)
}
