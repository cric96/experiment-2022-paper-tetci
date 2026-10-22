package it.unibo.learning.agents.offline

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, Step, Trajectory}
import it.unibo.learning.network.NeuralNetworkRL.SpatialTemporal

/** A logged step relabelled for a target trade-off weight `beta`: both states are conditioned on `beta`, so that a
  * single β-conditioned policy can be trained for every trade-off from the same experience.
  *
  * @param intensity
  *   the instability intensity of the step's state, which determines the counterfactual reward for `beta`
  * @param discount
  *   the semi-Markov discount γ^Δt of the step, where Δt is the wake-up period it chose
  */
final case class RelabelledStep(
    step: Step,
    state: Array[Float],
    nextState: Array[Float],
    beta: Double,
    intensity: Double,
    isTerminal: Boolean,
    discount: Float
) {
  def isUnstable: Boolean = intensity > 0.0

  def terminalFlag: Float = if (isTerminal) 1.0f else 0.0f
}

/** Hindsight relabelling of the trajectories collected with one trade-off weight for many others. */
object Relabelling {

  /** The trade-off weights every step is relabelled for, besides the learner's own one. */
  val CounterfactualWeights: Seq[Double] = Seq(0.5, 0.8, 0.95, 0.99, 1.0)

  /** How many times the learner's own weight is repeated, to oversample the trade-off it was configured with. */
  val OwnWeightRepetitions: Int = 2

  /** Wake-up periods shorter than this are discounted as if they lasted this long. */
  val MinDiscountPeriod: Double = 0.5

  def targetWeights(ownWeight: Double): Seq[Double] = Seq.fill(OwnWeightRepetitions)(ownWeight) ++ CounterfactualWeights

  /** Relabels every step of `trajectory` (encoded with `encode`, whose β feature is at `betaFeatureIndex`) for every
    * target weight; the last step is terminal and points to itself.
    */
  def relabel(
      trajectory: Trajectory,
      ownWeight: Double,
      gamma: Double,
      encode: AgentState => Array[Float],
      betaFeatureIndex: Int
  ): Seq[RelabelledStep] = {
    val states = trajectory.map(step => encode(step.state)).toIndexedSeq
    trajectory.zipWithIndex.flatMap { case (step, index) =>
      val isTerminal = index == trajectory.size - 1
      val nextState = if (isTerminal) states(index) else states(index + 1)
      val intensity = SpatialTemporal.instabilityIntensity(step.state)
      val discount = math.pow(gamma, math.max(MinDiscountPeriod, step.action)).toFloat
      targetWeights(ownWeight).map { beta =>
        def conditioned(features: Array[Float]) = withBetaFeature(features, beta, betaFeatureIndex)
        RelabelledStep(step, conditioned(states(index)), conditioned(nextState), beta, intensity, isTerminal, discount)
      }
    }
  }

  private def withBetaFeature(features: Array[Float], beta: Double, betaFeatureIndex: Int): Array[Float] = {
    val conditioned = features.clone()
    conditioned(betaFeatureIndex) = TradeOff.betaFeature(beta)
    conditioned
  }
}
