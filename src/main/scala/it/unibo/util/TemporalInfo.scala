package it.unibo.util

object TemporalInfo {

  /** The sign (−1, 0, +1) of the change between consecutive values; non-finite values count as −1. A single value has a
    * stable (0) trend.
    */
  def computeDeltaTrend(info: Iterable[Double]): List[Double] =
    info
      .map {
        case x if !x.isFinite => -1
        case x => x
      }
      .sliding(2, 1)
      .toList
      .map(x =>
        if x.size == 1 then { 0.0 }
        else { (x.last - x.head).sign }
      )
}
