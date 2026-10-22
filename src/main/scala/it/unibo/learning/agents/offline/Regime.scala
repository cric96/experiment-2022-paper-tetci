package it.unibo.learning.agents.offline

import it.unibo.learning.encoding.SpatialTemporal.Feature

/** The operating regime of a node, read from its spatial-temporal features: used to colour the inspected states. */
enum Regime(val label: String) {

  /** The wave of change is crossing the node, which routes other nodes. */
  case BackboneWave extends Regime("Backbone_Wave")

  /** The wave of change is crossing the node, which routes no other node. */
  case LeafWave extends Regime("Leaf_Wave")

  /** The node is stable but its neighbourhood is changing: the wave is approaching. */
  case NeighbourEarlyWarning extends Regime("Neighbor_EarlyWarning")

  /** Nothing is changing around the node. */
  case SteadyState extends Regime("Steady_State")
}

object Regime {

  /** Trends whose magnitude is below this value are considered stable. */
  val TrendThreshold: Float = 0.01f

  /** Own changes below this magnitude are ignored. */
  val SelfChangeThreshold: Float = 0.05f

  /** Neighbour changes, ratios of changed neighbours and residuals below these values are ignored. */
  val NeighbourChangeThreshold: Float = 0.02f
  val ChangedRatioThreshold: Float = 0.01f
  val ResidualThreshold: Float = 0.05f

  /** A node with a children ratio above this value belongs to the backbone of the shortest-path tree. */
  val BackboneThreshold: Float = 0.15f

  def of(features: Array[Float]): Regime = {
    val isHit = math.abs(features(Feature.LatestTrend)) > TrendThreshold ||
      math.abs(features(Feature.PreviousTrend)) > TrendThreshold ||
      features(Feature.SelfChange) > SelfChangeThreshold
    val isWarned = features(Feature.NeighbourChange) > NeighbourChangeThreshold ||
      features(Feature.ChangedRatio) > ChangedRatioThreshold ||
      features(Feature.Residual) > ResidualThreshold
    val isBackbone = features(Feature.ChildrenRatio) > BackboneThreshold
    if isHit && isBackbone then BackboneWave
    else if isHit then LeafWave
    else if isWarned then NeighbourEarlyWarning
    else SteadyState
  }
}
