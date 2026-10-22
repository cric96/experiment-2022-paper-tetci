package it.unibo.inspection

import it.unibo.inspection.NodeView._
import it.unibo.util.Csv

/** What the snapshot exporters record of a node at a given time. */
final case class NodeView(
    id: Int,
    x: Double,
    y: Double,
    isSource: Boolean,
    output: Double,
    groundTruth: Double,
    period: Double,
    ticks: Int
) {
  def boundedOutput: Double = bounded(output)

  def boundedGroundTruth: Double = bounded(groundTruth)

  def error: Double = math.abs(boundedOutput - boundedGroundTruth)

  /** Id, position, source flag, output, ground truth, wake-up period and error, as exported. */
  def columns: Seq[String] = Seq(
    id.toString,
    Csv.number(x, CoordinateDecimals),
    Csv.number(y, CoordinateDecimals),
    if (isSource) "1" else "0",
    Csv.number(boundedOutput, ValueDecimals),
    Csv.number(boundedGroundTruth, ValueDecimals),
    Csv.number(period, PeriodDecimals),
    Csv.number(error, ValueDecimals)
  )
}

object NodeView {

  /** Value exported for non-finite outputs (unreachable nodes): larger than any distance in the arena. */
  val UnboundedOutput: Double = 600.0

  val CoordinateDecimals: Int = 2
  val ValueDecimals: Int = 2
  val PeriodDecimals: Int = 4

  /** Header of [[NodeView.columns]]. */
  val Header: Seq[String] = Seq("nodeId", "x", "y", "isSource", "gradient", "groundTruth", "nextWakeUp", "error")

  private def bounded(value: Double): Double = if (value.isFinite) value else UnboundedOutput
}

/** A link of the network between two nodes, with their positions. */
final case class Link(from: Int, to: Int, fromPosition: (Double, Double), toPosition: (Double, Double)) {
  def columns: Seq[String] =
    Seq(from.toString, to.toString) ++
      Seq(fromPosition._1, fromPosition._2, toPosition._1, toPosition._2).map(
        Csv.number(_, NodeView.CoordinateDecimals)
      )
}
