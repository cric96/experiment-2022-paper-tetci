package it.unibo.learning.encoding

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.encoding.NeighbourhoodSignals.*
import it.unibo.util.TemporalInfo

/** How urgently a node should run its next round, I(s) ∈ [0, 1]: maximal when its own output just changed, partial (and
  * amplified for backbone nodes) when its neighbourhood is changing, and decaying after a change.
  */
object InstabilityIntensity:

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

  def of(state: AgentState): Double =
    val ownTrends = TemporalInfo.computeDeltaTrend(state.ownOutputs.takeRight(OwnHistoryLength))
    val roleMultiplier = LeafRoleMultiplier + BackboneRoleBonus * downstreamChildrenRatio(state)
    val intensity =
      if ownTrends.takeRight(RecentTrends).exists(_ != 0.0) then 1.0
      else
        neighbourhoodSignal(state).fold(if ownTrends.exists(_ != 0.0) then Cooldown * roleMultiplier else 0.0) { signal =>
          math.min(1.0, math.max(NeighbourhoodFloor, signal) * roleMultiplier)
        }
    math.min(1.0, intensity)

  /** The strongest sign that the neighbourhood is still changing, if any is above its threshold. */
  private def neighbourhoodSignal(state: AgentState): Option[Double] =
    val change = neighbourhoodChange(state, localScale(state))
    val residual = bellmanFordResidual(state)
    val isChanging =
      change.maxMagnitude > ChangeThreshold || change.changedRatio > 0.0 || residual > ResidualThreshold
    Option.when(isChanging)(math.max(change.maxMagnitude, math.max(change.changedRatio, residual)))
end InstabilityIntensity
