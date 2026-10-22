package it.unibo.learning.network

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.AgentState.{NeighborInfo, Snapshot}
import it.unibo.util.TemporalInfo

/** State encoders that turn an [[AgentState]] into the feature vectors fed to the networks. */
object NeuralNetworkRL {

  /** Number of trends of the node's own output fed to the networks (latest and previous). */
  val TrendFeatures: Int = 2

  object Historical {

    /** The last `snapshots` trends (−1, 0, +1) of the node's own output, most recent first, padded with zeros. */
    def encodeHistory(state: AgentState, snapshots: Int): Array[Double] = {
      val outputs = state.ownOutputs.replaceInfinite().takeRight(snapshots + 1)
      TemporalInfo.computeDeltaTrend(outputs).reverse.take(snapshots).padTo(snapshots, 0.0).toArray
    }
  }

  /** Spatial-temporal encoder with topological role and trade-off conditioning. Features, all in [−1, 1]:
    *   - 0: latest trend of the node's output (−1, 0, +1)
    *   - 1: previous trend of the node's output (−1, 0, +1)
    *   - 2: magnitude of the last change of the node's output, tanh(|g_i(t) − g_i(t−1)| / scale)
    *   - 3: largest change of a neighbour's output, tanh(max_j |g_j(t) − g_j(t−1)| / scale)
    *   - 4: fraction of neighbours whose output changed in the last round
    *   - 5: local Bellman-Ford residual (see [[bellmanFordResidual]])
    *   - 6: fraction of neighbours routed through the node in the shortest-path tree (backbone vs. leaf)
    *   - 7: shortest wake-up period among the neighbours, divided by the longest admissible period
    *   - 8: the trade-off weight β (see [[TradeOff.betaFeature]])
    */
  object SpatialTemporal {
    val FeatureDim: Int = 9

    /** Positions of the features in the encoded vector. */
    object Feature {
      val LatestTrend: Int = 0
      val PreviousTrend: Int = 1
      val SelfChange: Int = 2
      val NeighbourChange: Int = 3
      val ChangedRatio: Int = 4
      val Residual: Int = 5
      val ChildrenRatio: Int = 6
      val ShortestNeighbourPeriod: Int = 7
      val Beta: Int = 8
    }

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

    /** Constants of [[instabilityIntensity]]. */
    object Intensity {

      /** Number of own outputs inspected to detect recent changes of the node. */
      val OwnHistoryLength: Int = 4

      /** A change of the node in its last `RecentTrends` trends makes it fully unstable. */
      val RecentTrends: Int = 2

      /** Neighbourhood changes below this normalized magnitude are ignored. */
      val ChangeThreshold: Double = 1e-3

      /** Bellman-Ford residuals below this value are ignored. */
      val ResidualThreshold: Double = 0.05

      /** Minimum intensity when the neighbourhood is changing. */
      val NeighbourhoodFloor: Double = 0.45

      /** Intensity when the node's own output changed in the window, but not recently. */
      val Cooldown: Double = 0.35

      /** Role multiplier of a leaf (no children); a node whose neighbours are all children gets the whole bonus. */
      val LeafRoleMultiplier: Double = 0.75
      val BackboneRoleBonus: Double = 0.5
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

    /** Magnitude of the outputs around the node, used to normalize every difference of outputs. */
    def localScale(state: AgentState): Double =
      state.latestSnapshot
        .map(_.values.map(_.data).filter(isPlausible))
        .filter(_.nonEmpty)
        .fold(DefaultScale)(outputs => math.max(1.0, ScaleFraction * outputs.map(math.abs).max))

    private def isPlausible(output: Double): Boolean = output.isFinite && math.abs(output) < MaxPlausibleOutput

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

    /** Instability intensity I(s) ∈ [0, 1], combining own history, neighbourhood changes and topological role. */
    def instabilityIntensity(state: AgentState): Double = {
      val ownTrends = TemporalInfo.computeDeltaTrend(state.ownOutputs.takeRight(Intensity.OwnHistoryLength))
      val roleMultiplier = Intensity.LeafRoleMultiplier + Intensity.BackboneRoleBonus * downstreamChildrenRatio(state)
      val intensity =
        if (ownTrends.takeRight(Intensity.RecentTrends).exists(_ != 0.0)) {
          1.0
        } else {
          neighbourhoodSignal(state).fold(
            if (ownTrends.exists(_ != 0.0)) Intensity.Cooldown * roleMultiplier else 0.0
          ) { signal =>
            math.min(1.0, math.max(Intensity.NeighbourhoodFloor, signal) * roleMultiplier)
          }
        }
      math.min(1.0, intensity)
    }

    /** The strongest sign that the neighbourhood is still changing, if any is above its threshold. */
    private def neighbourhoodSignal(state: AgentState): Option[Double] = {
      val change = neighbourhoodChange(state, localScale(state))
      val residual = bellmanFordResidual(state)
      val isChanging = change.maxMagnitude > Intensity.ChangeThreshold || change.changedRatio > 0.0 ||
        residual > Intensity.ResidualThreshold
      Option.when(isChanging)(math.max(change.maxMagnitude, math.max(change.changedRatio, residual)))
    }

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

    def encode(state: AgentState, weight: Double, maxAction: Double): Array[Float] = {
      val scale = localScale(state)
      val trends = Historical.encodeHistory(state, TrendFeatures)
      val change = neighbourhoodChange(state, scale)
      Array(
        trends(0).toFloat,
        trends(1).toFloat,
        selfChangeMagnitude(state, scale).toFloat,
        change.maxMagnitude.toFloat,
        change.changedRatio.toFloat,
        bellmanFordResidual(state).toFloat,
        downstreamChildrenRatio(state).toFloat,
        shortestNeighbourPeriod(state, maxAction),
        TradeOff.betaFeature(weight)
      )
    }

    /** The shortest wake-up period among the neighbours over `maxAction` (1 if the node has no neighbours). */
    private def shortestNeighbourPeriod(state: AgentState, maxAction: Double): Float =
      state.latestSnapshot
        .flatMap(state.neighboursIn(_).values.map(_.oldAction).minOption)
        .fold(1.0f)(period => (period / maxAction).toFloat)
  }

  /** Star-graph encoder for the EdgeGAT networks: the ego node features followed by one slot of features for each of
    * the `MaxNeighbors` nearest neighbours (padded with idle slots).
    *
    * Ego features: latest and previous own trend, magnitude of the last own change, children ratio, β feature. Slot
    * features: sign and magnitude of the neighbour's last change, drift of the edge difference, relative order of the
    * outputs, normalized distance, normalized wake-up period.
    */
  object StarGraph {
    val SelfDim: Int = 5
    val MaxNeighbors: Int = 8
    val NeighDim: Int = 6
    val TotalDim: Int = SelfDim + MaxNeighbors * NeighDim

    /** Position of the β feature among the ego features. */
    val BetaFeature: Int = SelfDim - 1

    /** Distance that normalizes the neighbours' distances to [0, 1]. */
    val MaxDistance: Double = 100.0

    /** Slot of a missing neighbour: nothing changes and it sleeps as long as possible. */
    val IdleSlot: Seq[Float] = Seq(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f)

    /** What a neighbour slot is computed against: the previous snapshot and the node's own last two outputs. */
    final private case class Reference(previous: Snapshot, ownNow: Double, ownBefore: Double, scale: Double)

    def encode(state: AgentState, weight: Double, maxAction: Double): Array[Float] = {
      val scale = SpatialTemporal.localScale(state)
      (egoFeatures(state, weight, scale) ++ neighbourSlots(state, scale, maxAction).flatten).toArray
    }

    private def egoFeatures(state: AgentState, weight: Double, scale: Double): Seq[Float] = {
      val trends = Historical.encodeHistory(state, TrendFeatures)
      Seq(
        trends(0).toFloat,
        trends(1).toFloat,
        SpatialTemporal.selfChangeMagnitude(state, scale).toFloat,
        SpatialTemporal.downstreamChildrenRatio(state).toFloat,
        TradeOff.betaFeature(weight)
      )
    }

    private def neighbourSlots(state: AgentState, scale: Double, maxAction: Double): Seq[Seq[Float]] = {
      val current = state.latestSnapshot.getOrElse(Map.empty[Int, NeighborInfo])
      val previous = state.lastTwoSnapshots.fold(current) { case (previous, _) => previous }
      val reference = Reference(
        previous,
        ownNow = current.get(state.me).fold(0.0)(_.data),
        ownBefore = previous.get(state.me).fold(0.0)(_.data),
        scale
      )
      state
        .neighboursIn(current)
        .toList
        .sortBy { case (_, neighbour) => neighbour.distance }
        .take(MaxNeighbors)
        .map { case (id, neighbour) => slotOf(id, neighbour, reference, maxAction) }
        .padTo(MaxNeighbors, IdleSlot)
    }

    private def slotOf(id: Int, neighbour: NeighborInfo, reference: Reference, maxAction: Double): Seq[Float] = {
      val now = bounded(neighbour.data, reference.scale)
      val before = reference.previous.get(id).fold(now)(info => bounded(info.data, reference.scale))
      val change = now - before
      Seq(
        math.signum(change).toFloat,
        math.tanh(math.abs(change) / reference.scale).toFloat,
        edgeDrift(now, before, reference).toFloat,
        relativeOrder(now, reference).toFloat,
        math.min(1.0, neighbour.distance / MaxDistance).toFloat,
        (neighbour.oldAction / maxAction).toFloat
      )
    }

    private def bounded(output: Double, scale: Double): Double =
      if (output.isFinite) output else scale * SpatialTemporal.UnboundedChangeInScales

    private def edgeDrift(now: Double, before: Double, reference: Reference): Double =
      if (reference.ownNow.isFinite && reference.ownBefore.isFinite) {
        math.tanh(math.abs((reference.ownNow - now) - (reference.ownBefore - before)) / reference.scale)
      } else {
        0.0
      }

    private def relativeOrder(now: Double, reference: Reference): Double =
      if (reference.ownNow.isFinite) math.tanh((now - reference.ownNow) / reference.scale) else 0.0
  }
}
