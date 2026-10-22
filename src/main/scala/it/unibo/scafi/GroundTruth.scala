package it.unibo.scafi

import scala.annotation.tailrec
import scala.collection.immutable.Queue
import scala.collection.mutable

/** The values the building blocks converge to, computed centrally on the [[Topology]], used to measure the error of the
  * nodes. Nodes that cannot be reached are absent from the returned maps.
  */
object GroundTruth {

  /** Gradient: length of the shortest path from the nearest source (Dijkstra). */
  def shortestPaths(topology: Topology, sources: Set[Int]): Map[Int, Double] = {
    val queue = mutable.PriorityQueue.empty[(Double, Int)](using Ordering.by[(Double, Int), Double](-_._1))
    @tailrec def settle(distances: Map[Int, Double]): Map[Int, Double] =
      if (queue.isEmpty) {
        distances
      } else {
        val (distance, node) = queue.dequeue()
        if (distance > distanceOf(distances, node)) settle(distances)
        else
          settle(topology.neighboursOf(node).foldLeft(distances) { case (current, (neighbour, length)) =>
            val candidate = distance + length
            if (candidate < distanceOf(current, neighbour)) {
              queue.enqueue((candidate, neighbour))
              current.updated(neighbour, candidate)
            } else {
              current
            }
          })
      }
    val origins = sources.filter(topology.nodes.contains)
    origins.foreach(source => queue.enqueue((0.0, source)))
    settle(origins.map(_ -> 0.0).toMap)
  }

  /** Hop gradient: number of links from the nearest source (breadth-first search). */
  def hopDistances(topology: Topology, sources: Set[Int]): Map[Int, Double] = {
    @tailrec def visit(frontier: Queue[Int], distances: Map[Int, Double]): Map[Int, Double] =
      frontier.dequeueOption match {
        case None => distances
        case Some((node, rest)) =>
          val hops = distances(node) + 1.0
          val improved = topology.neighboursOf(node).map(_._1).filter(hops < distanceOf(distances, _))
          visit(rest ++ improved, distances ++ improved.map(_ -> hops))
      }
    val origins = sources.filter(topology.nodes.contains).toSeq
    visit(Queue.from(origins), origins.map(_ -> 0.0).toMap)
  }

  /** Collect: each node counts itself plus the counts of its children in the spanning tree of the `potentials`, where
    * the parent of a node is its neighbour with the lowest potential (then the lowest id).
    */
  def collectCounts(topology: Topology, potentials: Map[Int, Double]): Map[Int, Double] =
    if (potentials.isEmpty) {
      Map.empty
    } else {
      val potential = distanceOf(potentials, _: Int)
      val farthestFirst = topology.nodes.sortBy(node => -potential(node))(Ordering.Double.TotalOrdering)
      farthestFirst.foldLeft(topology.nodes.map(_ -> 1.0).toMap) { (counts, node) =>
        parentOf(topology, node, potential).fold(counts)(parent =>
          counts.updated(parent, counts(parent) + counts(node))
        )
      }
    }

  private def parentOf(topology: Topology, node: Int, potential: Int => Double): Option[Int] = {
    val own = potential(node)
    Option
      .when(own > 0.0 && own.isFinite)(topology.neighboursOf(node).map(_._1).filter(potential(_) < own))
      .flatMap(_.minByOption(neighbour => (potential(neighbour), neighbour))(lowestPotentialThenId))
  }

  private val lowestPotentialThenId: Ordering[(Double, Int)] =
    Ordering.Tuple2(Ordering.Double.TotalOrdering, Ordering.Int)

  private def distanceOf(distances: Map[Int, Double], node: Int): Double =
    distances.getOrElse(node, Double.PositiveInfinity)
}
