package it.unibo.learning.agents.offline

/** How an offline learner explores and trains across the episodes. During the first `explorationEpisodes` the nodes
  * explore with an ε that decays linearly from `initialEpsilon` to zero, and the learner trains for `stepsPerEpisode`
  * gradient steps at the end of each episode, `finalTrainingMultiplier` times longer at the end of the last one.
  * Afterwards the policy is frozen and only exploited.
  */
final case class TrainingSchedule(
    explorationEpisodes: Int,
    stepsPerEpisode: Int,
    finalTrainingMultiplier: Int,
    initialEpsilon: Double = TrainingSchedule.DefaultInitialEpsilon
):
  def isExploring(episode: Int): Boolean = episode < explorationEpisodes

  /** Whether the training session at the end of `episode` is the last one. */
  def isLastTraining(episode: Int): Boolean = episode == explorationEpisodes - 1

  def epsilonAt(episode: Int): Double =
    if isExploring(episode) then initialEpsilon - initialEpsilon * episode.toDouble / explorationEpisodes.toDouble
    else 0.0

  /** The number of gradient steps to perform at the end of `episode`, if the learner still trains. */
  def trainingStepsAfter(episode: Int): Option[Int] =
    if isLastTraining(episode) then Some(stepsPerEpisode * finalTrainingMultiplier)
    else Option.when(isExploring(episode))(stepsPerEpisode)

object TrainingSchedule:

  /** Probability of a random action in the first episode. */
  val DefaultInitialEpsilon: Double = 0.99
