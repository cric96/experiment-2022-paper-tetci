package it.unibo.scafi

/** How the episodes of an experiment are used: the policy is trained on the gradient during the first
  * `TrainingEpisodes`; afterwards (in the generalization scenario) it runs the target building block and is evaluated
  * on each of the `EvaluationWeights` in turn, for `EpisodesPerWeight` episodes each.
  */
object EvaluationProtocol:
  val TrainingEpisodes: Int = 10
  val EpisodesPerWeight: Int = 2
  val EvaluationWeights: Seq[Double] = Seq(0.50, 0.80, 0.95, 0.99, 1.00)

  def isTraining(episode: Int): Boolean = episode < TrainingEpisodes

  def blockAt(episode: Int, target: BuildingBlock): BuildingBlock =
    if isTraining(episode) then BuildingBlock.Gradient else target

  /** The trade-off weight under evaluation in `episode`, once training is over. */
  def evaluationWeightAt(episode: Int): Option[Double] =
    Option.unless(isTraining(episode)) {
      EvaluationWeights(math.min(EvaluationWeights.size - 1, (episode - TrainingEpisodes) / EpisodesPerWeight))
    }
