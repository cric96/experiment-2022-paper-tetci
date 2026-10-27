package it.unibo.inspection

/** Episode-local counters frozen at node failure, so removal does not erase completed work from the exports. */
final class FailedNodeMetrics:
  private var totals: Map[String, Double] = Map.empty
  private var count: Int = 0

  def failedNodes: Int = count

  def totalOf(molecule: String): Double = totals.getOrElse(molecule, 0.0)

  def record(values: Map[String, Double]): Unit =
    values.foreach { case (molecule, value) => totals = totals.updated(molecule, totalOf(molecule) + value) }
    count += 1

  def reset(): Unit =
    totals = Map.empty
    count = 0
