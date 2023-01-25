package it.unibo.util

object TemporalInfo {
  def computeDeltaTrend(info: Iterable[Double]): List[Double] =
    info
      .sliding(2, 1)
      .toList
      .map(x =>
        if (x.size == 1) { 0.0 }
        else { (x.last - x.head).sign }
      )
}
