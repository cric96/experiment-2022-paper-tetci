package it.unibo.learning.encoding

import it.unibo.learning.abstractions.AgentState
import it.unibo.util.TemporalInfo

/** The recent trends (−1, 0, +1) of the node's own output. */
object OwnTrends:

  /** The last `count` trends, most recent first, padded with zeros when the history is shorter. */
  def recent(state: AgentState, count: Int): Array[Double] =
    val outputs = state.ownOutputs.replaceInfinite().takeRight(count + 1)
    TemporalInfo.computeDeltaTrend(outputs).reverse.take(count).padTo(count, 0.0).toArray
