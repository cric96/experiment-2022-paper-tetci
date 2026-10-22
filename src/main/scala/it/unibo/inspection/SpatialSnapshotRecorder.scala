package it.unibo.inspection

import it.unibo.inspection.SpatialSnapshotRecorder.*
import it.unibo.util.Csv

import java.io.File
import scala.collection.mutable.ArrayBuffer

/** Records the state of every node over time, and the links of the network, for the spatial figures of a run with
  * trade-off `weight`.
  */
final class SpatialSnapshotRecorder(directory: File, weight: Double) {
  private val rows: ArrayBuffer[Seq[String]] = ArrayBuffer.empty
  private var links: Seq[Seq[String]] = Seq.empty

  /** Records `nodes` at `timeInEpisode`; the links are taken (once, each only in one direction) from `network`. */
  def record(timeInEpisode: Double, nodes: Seq[NodeView], network: => Seq[Link]): Unit = {
    if links.isEmpty then links = network.filter(link => link.from < link.to).map(_.columns)
    rows ++= nodes.map(node => (Csv.number(timeInEpisode, TimeDecimals) +: node.columns) :+ node.ticks.toString)
  }

  /** Writes the files, if something has been recorded. */
  def flush(): Unit =
    if rows.nonEmpty then {
      Csv.write(directory, s"spatial_snapshots_weight-$weight.csv", SnapshotHeader, rows)
      Csv.write(directory, s"spatial_edges_weight-$weight.csv", LinkHeader, links)
    }
}

object SpatialSnapshotRecorder {
  val TimeDecimals: Int = 2
  val SnapshotHeader: Seq[String] = ("tEp" +: NodeView.Header) :+ "ticks"
  val LinkHeader: Seq[String] = Seq("u", "v", "xU", "yU", "xV", "yV")
}
