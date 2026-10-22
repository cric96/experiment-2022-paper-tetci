package it.unibo.learning.encoding

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.AgentState.{NeighborInfo, Snapshot}

/** Signals that tell how much, and how, the outputs around a node are changing; differences of outputs are normalized
  * by the [[localScale]] of the neighbourhood.
  */
object NeighbourhoodSignals {

  /** Scale used when the node has no finite output to compare with. */
  val DefaultScale: Double = 50.0

  /** The local scale is this fraction of the largest finite output in the neighbourhood. */
  val ScaleFraction: Double = 0.2

  /** Outputs larger than this are considered unbounded (e.g. a gradient with no source). */
  val MaxPlausibleOutput: Double = 1e6

  /** A change from or to a non-finite output counts as this many local scales. */
  val UnboundedChangeInScales: Double = 2.0

  /** A neighbour j is a child of i if |g_j − (g_i + d_ij)| ≤ max(MinChildTolerance, ChildToleranceRatio · d_ij). */
  val MinChildTolerance: Double = 8.0
  val ChildToleranceRatio: Double = 0.2

  /** How the neighbours' outputs changed between the last two rounds. */
  final case class NeighbourhoodChange(maxMagnitude: Double, changedRatio: Double)

  object NeighbourhoodChange {
    val None: NeighbourhoodChange = NeighbourhoodChange(0.0, 0.0)
  }

  /** Magnitude of the outputs around the node, used to normalize every difference of outputs. */
  def localScale(state: AgentState): Double =
    state.latestSnapshot
      .map(_.values.map(_.data).filter(isPlausible))
      .filter(_.nonEmpty)
      .fold(DefaultScale)(outputs => math.max(1.0, ScaleFraction * outputs.map(math.abs).max))

  private def isPlausible(output: Double): Boolean = output.isFinite && math.abs(output) < MaxPlausibleOutput

  /** `output` itself if finite, otherwise a value [[UnboundedChangeInScales]] local scales large. */
  def bounded(output: Double, scale: Double): Double = if (output.isFinite) output else scale * UnboundedChangeInScales

  /** Magnitude of the last change of the node's own output, in [0, 1]. */
  def selfChangeMagnitude(state: AgentState, scale: Double): Double = {
    val recent = state.ownOutputs.takeRight(2).replaceInfinite()
    if (recent.size >= 2) math.tanh(math.abs(recent.last - recent.head) / scale) else 0.0
  }

  /** Largest change (normalized in [0, 1]) and fraction of the neighbours whose output changed in the last round. */
  def neighbourhoodChange(state: AgentState, scale: Double): NeighbourhoodChange =
    state.lastTwoSnapshots.fold(NeighbourhoodChange.None) { case (previous, current) =>
      val neighbours = state.neighboursIn(current)
      if (neighbours.isEmpty) {
        NeighbourhoodChange.None
      } else {
        val changes = outputChanges(previous, neighbours, scale)
        NeighbourhoodChange(
          maxMagnitude = math.tanh(changes.foldLeft(0.0)(math.max) / scale),
          changedRatio = changes.size.toDouble / neighbours.size.toDouble
        )
      }
    }

  /** The absolute changes of the neighbours whose output changed since the `previous` snapshot. */
  private def outputChanges(previous: Snapshot, neighbours: Snapshot, scale: Double): List[Double] =
    neighbours.toList.flatMap { case (id, now) =>
      previous.get(id).filter(_.data != now.data).map(before => outputChange(before.data, now.data, scale))
    }

  private def outputChange(before: Double, now: Double, scale: Double): Double =
    if (before.isFinite && now.isFinite) math.abs(now - before) else scale * UnboundedChangeInScales

  /** How much the differences between the node's output and its neighbours' drifted in the last round, in [0, 1]: a
    * non-zero value means that the local Bellman-Ford equation is not yet stable.
    */
  def bellmanFordResidual(state: AgentState): Double =
    state.lastTwoSnapshots.fold(0.0) { case (previous, current) =>
      val ownNow = current.get(state.me).fold(0.0)(_.data)
      val ownBefore = previous.get(state.me).fold(0.0)(_.data)
      val neighbours = state.neighboursIn(current)
      if (neighbours.isEmpty || !ownNow.isFinite || !ownBefore.isFinite) 0.0
      else math.tanh(maxEdgeDrift(previous, neighbours, ownNow, ownBefore) / localScale(state))
    }

  private def maxEdgeDrift(previous: Snapshot, neighbours: Snapshot, ownNow: Double, ownBefore: Double): Double =
    neighbours.foldLeft(0.0) { case (drift, (id, now)) =>
      previous.get(id) match {
        case Some(before) if now.data.isFinite && before.data.isFinite =>
          math.max(drift, math.abs((ownNow - now.data) - (ownBefore - before.data)))
        case _ => drift
      }
    }

  /** Fraction of the neighbours that route their shortest path through this node (g_j ≈ g_i + d_ij). */
  def downstreamChildrenRatio(state: AgentState): Double = {
    val ratio = for {
      current <- state.latestSnapshot
      own <- current.get(state.me).map(_.data) if own.isFinite
      neighbours = state.neighboursIn(current) if neighbours.nonEmpty
    } yield neighbours.count { case (_, neighbour) => isChild(own, neighbour) }.toDouble / neighbours.size.toDouble
    ratio.getOrElse(0.0)
  }

  private def isChild(own: Double, neighbour: NeighborInfo): Boolean = {
    val tolerance = math.max(MinChildTolerance, ChildToleranceRatio * neighbour.distance)
    neighbour.data.isFinite && neighbour.data > own && math.abs(
      neighbour.data - (own + neighbour.distance)
    ) <= tolerance
  }
}
