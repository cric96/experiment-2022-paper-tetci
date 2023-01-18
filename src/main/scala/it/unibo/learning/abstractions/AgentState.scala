package it.unibo.learning.abstractions

import it.unibo.learning.abstractions.AgentState.NeighborInfo

import scala.collection.immutable.Queue

/**
 */
case class AgentState(me: Int, neighborhoodOutput: Queue[Map[Int, NeighborInfo]], neighborhoodSensing: Queue[Map[Int, NeighborInfo]], contextual: Contextual)

object AgentState {
  case class NeighborInfo(data: Double, distance: Double)
}
