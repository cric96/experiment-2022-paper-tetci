package it.unibo.learning.agents.offline

/** How an offline learner explores and trains across the episodes. During the first `explorationEpisodes` the nodes
  * explore with an ε that decays linearly from `initialEpsilon` to zero, and the learner trains for `stepsPerEpisode`
  * gradient steps at the end of each episode; at the end of the last exploration episode it trains one last time,
  * `finalTrainingMultiplier` times longer. Afterwards the policy is only exploited.
  */
final case class TrainingSchedule(
    explorationEpisodes: Int,
    stepsPerEpisode: Int,
    finalTrainingMultiplier: Int,
    initialEpsilon: Double = TrainingSchedule.DefaultInitialEpsilon
) {
  def isExploring(episode: Int): Boolean = episode < explorationEpisodes

  def epsilonAt(episode: Int): Double =
    if isExploring(episode) then initialEpsilon - initialEpsilon * episode.toDouble / explorationEpisodes.toDouble
    else 0.0

  /** The number of gradient steps to perform at the end of `episode`, if the learner still trains. */
  def trainingStepsAfter(episode: Int): Option[Int] =
    if isExploring(episode) then Some(stepsPerEpisode)
    else Option.when(episode == explorationEpisodes)(stepsPerEpisode * finalTrainingMultiplier)
}

object TrainingSchedule {

  /** Probability of a random action in the first episode. */
  val DefaultInitialEpsilon: Double = 0.99
}
