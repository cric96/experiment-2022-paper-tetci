package it.unibo.learning.agents.offline

/** The synthetic states on which the response surface of a policy is evaluated: a grid of own and neighbour change
  * magnitudes, for a few children ratios, with the remaining features derived from them as they co-occur in practice.
  */
object PolicySurface:

  /** Number of intervals of the grid of change magnitudes in [0, 1]. */
  val Resolution: Int = 25

  /** Children ratios of a leaf and of a typical backbone node. */
  val ChildrenRatios: Seq[Float] = Seq(0.0f, 0.5f)

  /** Changes above this magnitude count as a change (own trend set, neighbours marked as changed). */
  val ActivityThreshold: Float = 0.05f

  /** Fraction of changed neighbours and Bellman-Ford residual per unit of neighbour change. */
  val ChangedRatioPerChange: Float = 0.8f
  val ResidualPerChange: Float = 0.5f

  /** Above this neighbour change the neighbours are busy and wake up fast, otherwise they sleep. */
  val BusyNeighbourhoodThreshold: Float = 0.1f
  val BusyNeighbourPeriod: Float = 0.3f
  val IdleNeighbourPeriod: Float = 0.8f

  final case class SurfacePoint(selfChange: Float, neighbourChange: Float, childrenRatio: Float):
    def latestTrend: Float = if selfChange > ActivityThreshold then 1.0f else 0.0f

    def changedRatio: Float =
      if neighbourChange > ActivityThreshold then math.min(1.0f, neighbourChange * ChangedRatioPerChange) else 0.0f

    def residual: Float = neighbourChange * ResidualPerChange

    def shortestNeighbourPeriod: Float =
      if neighbourChange > BusyNeighbourhoodThreshold then BusyNeighbourPeriod else IdleNeighbourPeriod

    /** The spatial-temporal features of the point (the previous trend is always stable). */
    def features(betaFeature: Float): Array[Float] =
      Array(
        latestTrend, 0.0f, selfChange, neighbourChange, changedRatio, residual, childrenRatio, shortestNeighbourPeriod,
        betaFeature
      )

  val points: Seq[SurfacePoint] =
    val grid = (0 to Resolution).map(_ / Resolution.toFloat)
    for
      selfChange <- grid
      neighbourChange <- grid
      childrenRatio <- ChildrenRatios
    yield SurfacePoint(selfChange, neighbourChange, childrenRatio)
end PolicySurface
