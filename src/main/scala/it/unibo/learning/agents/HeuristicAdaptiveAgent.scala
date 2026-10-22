package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.{AgentState, ReplayBuffer}
import it.unibo.util.TemporalInfo

import scala.util.Random

/** Hand-crafted Adaptive Heuristic Baseline (AIMD / Local Trend Thresholding):
  * Wakes up fast when the node's own output changes, ramps up on the next step, and sleeps at steadyAct when stable.
  */
class HeuristicAdaptiveAgent(
    actionSpace: Seq[Double],
    weight: Double = 0.95
) extends Learner {
  private val minAct: Double = actionSpace.min
  private val maxAct: Double = actionSpace.max
  private val steadyAct: Double = minAct + (maxAct - minAct) * math.max(0.45, 1.0 - 0.5 * weight)

  override def policyForWeight(customWeight: Double): AgentState => Double = { state =>
    val energyW = math.max(0.01, 1.0 - customWeight)
    val uSteady = energyW / (energyW + 0.06 * customWeight)
    val cSteady = minAct + (maxAct - minAct) * uSteady
    val cFast = math.max(minAct, minAct + (1.0 - customWeight) * 2.5)
    val myOutputs = state.neighborhoodOutput.flatMap(_.get(state.me)).takeRight(3).map(_.data)
    val trends = TemporalInfo.computeDeltaTrend(myOutputs)
    if (trends.lastOption.exists(_ != 0.0)) {
      cFast
    } else if (trends.exists(_ != 0.0)) {
      (cFast + cSteady) * 0.55
    } else {
      cSteady
    }
  }

  override def policy: AgentState => Double = policyForWeight(weight)

  override def store(where: String): Unit = {}

  override def load(where: String): AgentState => Double = policy

  override def update(batch: Seq[ReplayBuffer.Experience]): Unit = {}

  override def injectRandom(random: Random): Unit = {}

  override def injectCentralAgent(agent: CentralAgent[_, _]): Unit = {}
}
