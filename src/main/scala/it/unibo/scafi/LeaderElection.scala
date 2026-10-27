package it.unibo.scafi

/** Bounded priority claims implementing the sparse-choice rule of the paper. The preferred candidate is the device
  * carrying the source flag; all other devices remain candidates.
  */
object LeaderElection:
  /** Maximum distance travelled by a claim, in metres. */
  val Grain: Double = 400.0

  /** A candidate's identity, priority boost and accumulated path length. */
  final case class Claim(candidate: Int, preferred: Boolean, distance: Double):
    def isLeader(node: Int): Boolean = candidate == node && distance == 0.0

  def ownClaim(node: Int, preferred: Boolean): Claim = Claim(node, preferred, 0.0)

  /** Highest priority wins (boost first, then lowest id); shortest distance breaks equal-priority ties. */
  def choose(node: Int, preferred: Boolean, neighbours: Iterable[(Claim, Double)]): Claim =
    val received = neighbours
      .map { case (claim, length) => claim.copy(distance = claim.distance + length) }
      .filter(_.distance <= Grain)
    (received.toSeq :+ ownClaim(node, preferred)).maxBy(claim =>
      (if claim.preferred then 1 else 0, -claim.candidate, -claim.distance)
    )

  /** Central reference independent of the distributed round order: process candidates by decreasing priority,
    * suppressing the devices reached within the grain through the still-unclaimed subgraph. A suppressed device relays
    * the winning claim, so lower-priority claims cannot cross that region.
    */
  def reference(topology: Topology, preferred: Set[Int]): Map[Int, Double] =
    val candidates = topology.nodes.sortBy(node => (if preferred.contains(node) then 0 else 1, node))
    val (_, leaders) = candidates.foldLeft((topology.nodes.toSet, Set.empty[Int])) {
      case ((available, elected), candidate) =>
        if !available.contains(candidate) then (available, elected)
        else
          val remaining = Topology(
            topology.nodes.filter(available.contains),
            topology.neighbours.view
              .filterKeys(available.contains)
              .mapValues(_.filter(link => available.contains(link._1)))
              .toMap
          )
          val reached = GroundTruth
            .shortestPaths(remaining, Set(candidate))
            .collect {
              case (node, distance) if distance <= Grain => node
            }
            .toSet
          (available -- reached, elected + candidate)
    }
    topology.nodes.map(node => node -> (if leaders.contains(node) then 1.0 else 0.0)).toMap

  /** S followed by multi-source G: each node receives the id of its nearest elected leader. Distances are computed on
    * the full surviving graph, not the priority-claim regions.
    */
  def broadcastIds(topology: Topology, preferred: Set[Int]): Map[Int, Double] =
    val leaders = reference(topology, preferred).collect { case (node, flag) if flag == 1.0 => node }.toSeq.sorted
    val distances = leaders.map(leader => leader -> GroundTruth.shortestPaths(topology, Set(leader)))
    topology.nodes.map { node =>
      val leader = distances
        .minByOption { case (candidate, paths) =>
          (paths.getOrElse(node, Double.PositiveInfinity), candidate)
        }
        .map(_._1)
        .getOrElse(node)
      node -> leader.toDouble
    }.toMap
end LeaderElection
