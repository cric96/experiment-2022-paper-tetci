package it.unibo.learning.agents

import it.unibo.learning.abstractions.{DecayReference, ReplayBuffer}
import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Random

class QLearningSpec extends AnyFlatSpec with Matchers:
  private val actions = List(1.0, 2.0, 3.0, 4.0, 5.0)

  private def seededLearner(): QLearning =
    val learner =
      QLearning(actions, DecayReference.constant(0.2), DecayReference.constant(0.5), 0.9, 3)
    learner.injectRandom(new Random(7))
    learner

  private def threeRoundsOfDecisions(learner: QLearning): Seq[Double] =
    Seq.fill(3)(fixtures.map(learner.policy)).flatten

  "The tabular state" should "be the own trends followed by whether a neighbour changed" in {
    fixtures.map(seededLearner().encodeState) shouldBe
      Seq(List(0.0, -1.0, 0.0), List(0.0, 1.0), List(0.0, 0.0), List(0.0, 0.0))
  }

  "An untrained Q-Learning policy" should "keep its seeded epsilon-greedy choices" in {
    threeRoundsOfDecisions(seededLearner()) shouldBe
      Seq(1.0, 5.0, 1.0, 3.0, 1.0, 4.0, 3.0, 2.0, 5.0, 2.0, 5.0, 3.0)
  }

  "A trained Q-Learning policy" should "keep its seeded choices" in {
    val learner = seededLearner()
    val batch = fixtures.zip(fixtures.tail).zipWithIndex.map { case ((state, next), index) =>
      ReplayBuffer.Experience(state, index, (index + 1).toDouble, -0.1 * index, next)
    }
    (1 to 5).foreach(_ => learner.update(batch))
    threeRoundsOfDecisions(learner) shouldBe Seq(1.0, 5.0, 4.0, 5.0, 1.0, 4.0, 4.0, 2.0, 5.0, 4.0, 5.0, 4.0)
  }

  /** A learner that always explores while learning, and that has learnt to prefer `action` in `stableBackbone`. */
  private def learnerPreferring(action: Int): QLearning =
    val learner = QLearning(actions, DecayReference.constant(1.0), DecayReference.constant(0.5), 0.9, 3)
    learner.update(Seq(ReplayBuffer.Experience(stableBackbone, action, actions(action), 1.0, isolated)))
    learner

  private def afterLearningEpisodes(learner: QLearning): QLearning =
    (1 to QLearning.LearningEpisodes).foreach(_ => learner.endEpisode(Seq.empty))
    learner

  "A Q-Learning policy after the learning episodes" should "only exploit the Q table" in {
    val learner = afterLearningEpisodes(learnerPreferring(action = 2))
    Seq.fill(20)(learner.policy(stableBackbone)).distinct shouldBe Seq(actions(2))
  }

  it should "no longer update the Q table" in {
    val learner = afterLearningEpisodes(learnerPreferring(action = 2))
    learner.update(Seq.fill(10)(ReplayBuffer.Experience(stableBackbone, 4, actions(4), 10.0, isolated)))
    learner.policy(stableBackbone) shouldBe actions(2)
  }
end QLearningSpec
