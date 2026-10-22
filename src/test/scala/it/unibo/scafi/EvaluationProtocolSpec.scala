package it.unibo.scafi

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EvaluationProtocolSpec extends AnyFlatSpec with Matchers {
  private val lastTrainingEpisode = EvaluationProtocol.TrainingEpisodes - 1

  "The protocol" should "train on the gradient, then run the target building block" in {
    EvaluationProtocol.blockAt(lastTrainingEpisode, BuildingBlock.Collect) shouldBe BuildingBlock.Gradient
    EvaluationProtocol.blockAt(lastTrainingEpisode + 1, BuildingBlock.Collect) shouldBe BuildingBlock.Collect
  }

  it should "evaluate each weight for a few episodes after training, then keep the last one" in {
    val firstEvaluation = EvaluationProtocol.TrainingEpisodes
    EvaluationProtocol.evaluationWeightAt(lastTrainingEpisode) shouldBe None
    (0 until EvaluationProtocol.EpisodesPerWeight).foreach { offset =>
      EvaluationProtocol.evaluationWeightAt(firstEvaluation + offset) shouldBe Some(0.5)
    }
    EvaluationProtocol.evaluationWeightAt(firstEvaluation + EvaluationProtocol.EpisodesPerWeight) shouldBe Some(0.8)
    EvaluationProtocol.evaluationWeightAt(firstEvaluation + 100) shouldBe Some(1.0)
  }
}
