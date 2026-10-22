package it.unibo.learning.agents

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.agents.HeuristicAdaptiveAgent.*
import it.unibo.util.TemporalInfo

/** Hand-crafted adaptive baseline (local trend thresholding): a node wakes up fast as soon as its own output changes,
  * ramps up in the following rounds and sleeps at the steady-state period while its output is stable. Both periods
  * depend on the trade-off weight β.
  */
final class HeuristicAdaptiveAgent(actionSpace: Seq[Double], weight: Double = DefaultWeight) extends Learner {
  private val fastest: Double = actionSpace.min
  private val slowest: Double = actionSpace.max

  override def policy: AgentState => Double = policyForWeight(weight)

  override def policyForWeight(customWeight: Double): AgentState => Double = { state =>
    val trends = TemporalInfo.computeDeltaTrend(state.ownOutputs.takeRight(HistoryLength))
    if trends.lastOption.exists(_ != 0.0) then fastPeriod(customWeight)
    else if trends.exists(_ != 0.0) then (fastPeriod(customWeight) + steadyPeriod(customWeight)) * RampFactor
    else steadyPeriod(customWeight)
  }

  /** Period right after a change: the fastest one when only convergence matters, slower as β decreases. */
  private def fastPeriod(beta: Double): Double = math.max(fastest, fastest + (1.0 - beta) * FastPeriodSlope)

  /** Period of a stable node: the steady-state share of the energy objective of the action range. */
  private def steadyPeriod(beta: Double): Double = {
    val energy = math.max(MinEnergyWeight, 1.0 - beta)
    fastest + (slowest - fastest) * (energy / (energy + TradeOff.SteadyStateIntensity * beta))
  }
}

object HeuristicAdaptiveAgent {
  val DefaultWeight: Double = 0.95

  /** Number of own outputs inspected, i.e. the last two trends. */
  val HistoryLength: Int = 3

  /** Lower bound of the energy weight 1 − β in the steady period. */
  val MinEnergyWeight: Double = 0.01

  /** How much the period right after a change grows for each unit of energy weight 1 − β. */
  val FastPeriodSlope: Double = 2.5

  /** While ramping up, the period is this fraction of the sum of the fast and of the steady period. */
  val RampFactor: Double = 0.55
}
