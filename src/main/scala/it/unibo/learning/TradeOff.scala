package it.unibo.learning

/** The convergence/energy trade-off that every policy optimizes, controlled by the weight β ∈ [0, 1]: β = 1 means that
  * only convergence matters, lower values make slow (energy-saving) rounds more and more attractive.
  *
  * The wake-up period is normalized as u ∈ [0, 1] (0 = fastest admissible period): convergence is penalized by u²
  * (sleeping while the field changes) and energy by (1 − u)² (waking up while nothing changes).
  */
object TradeOff {

  /** Lower bound of the energy weight 1 − β, so that energy is never completely ignored, even for β = 1. */
  val MinEnergyWeight: Double = 0.005

  /** Residual convergence pressure of a node in steady state (instability intensity equal to zero). */
  val SteadyStateIntensity: Double = 0.06

  /** Gain of the instability intensity in the online reward (intensity 1 gives a convergence pressure of 1). */
  val OnlineIntensityGain: Double = 1.0 - SteadyStateIntensity

  /** Gain of the instability intensity in the counterfactual rewards used to train the offline learners. */
  val OfflineIntensityGain: Double = 8.0

  /** Normalized weights of the two objectives; they sum to one. */
  final case class ObjectiveWeights(convergence: Double, energy: Double)

  def energyWeight(beta: Double): Double = math.max(MinEnergyWeight, 1.0 - beta)

  /** β as fed to the networks: the steady-state share of the energy objective, mapped from [0, 1] to [−1, 1]. */
  def betaFeature(beta: Double): Float = {
    val energy = energyWeight(beta)
    (2.0 * (energy / (energy + SteadyStateIntensity * beta)) - 1.0).toFloat
  }

  /** The wake-up `period` mapped to [0, 1] within [`min`, `max`], clamped outside. */
  def normalizedPeriod(period: Double, min: Double, max: Double): Double =
    math.max(0.0, math.min(1.0, (period - min) / (max - min)))

  /** Normalized objective weights used by the offline learners, given the node's instability `intensity` ∈ [0, 1]. */
  def objectiveWeights(intensity: Double, beta: Double): ObjectiveWeights = {
    val energy = energyWeight(beta)
    val convergence = beta * (SteadyStateIntensity + OfflineIntensityGain * intensity)
    val total = convergence + energy
    ObjectiveWeights(convergence / total, energy / total)
  }

  /** Counterfactual reward used to train the offline learners for any β, from logged (intensity, period) pairs. */
  def offlineReward(intensity: Double, period: Double, beta: Double, min: Double, max: Double): Double = {
    val u = normalizedPeriod(period, min, max)
    val weights = objectiveWeights(intensity, beta)
    -weights.convergence * (u * u) - weights.energy * ((1.0 - u) * (1.0 - u))
  }

  /** Reward computed by the nodes during the simulation (used by Q-Learning and reported as `reward`). */
  def onlineReward(intensity: Double, period: Double, beta: Double, min: Double, max: Double): Double = {
    val u = normalizedPeriod(period, min, max)
    -beta * (SteadyStateIntensity + OnlineIntensityGain * intensity) * (u * u) -
      energyWeight(beta) * ((1.0 - u) * (1.0 - u))
  }
}
