package it.unibo.scafi

import it.unibo.scafi.BuildingBlock.*
import it.unibo.scafi.LeaderElection as SparseElection

/** The ground truths of every building block for one configuration of the network (its topology and sources). */
final case class GroundTruths(
    sources: Set[Int],
    gradient: Map[Int, Double],
    hops: Map[Int, Double],
    collectCounts: Map[Int, Double],
    leaderIds: Map[Int, Double]
):

  /** The value `block` should converge to in `node`: infinite (gradients) or 1 (collect) for unreachable nodes. */
  def of(node: Int, block: BuildingBlock): Double = block match
    case Gradient => gradient.getOrElse(node, Double.PositiveInfinity)
    case HopGradient => hops.getOrElse(node, Double.PositiveInfinity)
    case Broadcast => sources.headOption.fold(0.0)(BuildingBlock.broadcastPayloadOf)
    case Collect => collectCounts.getOrElse(node, 1.0)
    case LeaderElection => leaderIds.getOrElse(node, 0.0)

object GroundTruths:
  def of(topology: Topology, sources: Set[Int]): GroundTruths =
    val gradient = GroundTruth.shortestPaths(topology, sources)
    GroundTruths(
      sources,
      gradient,
      GroundTruth.hopDistances(topology, sources),
      GroundTruth.collectCounts(topology, gradient),
      SparseElection.broadcastIds(topology, sources)
    )
