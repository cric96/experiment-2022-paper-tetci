package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDManager
import it.unibo.learning.abstractions.{AgentState, Step, Trajectory}
import it.unibo.testing.AgentStates
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EpisodicOfflineLearnerSpec extends AnyFlatSpec with Matchers:
  private val schedule = TrainingSchedule(explorationEpisodes = 2, stepsPerEpisode = 3, finalTrainingMultiplier = 2)
  private val trajectory: Trajectory = Seq(Step(AgentStates.isolated, 0, 1.0, 0.0))

  /** Records one transition per step and counts the gradient steps and the training sessions. */
  final private class CountingLearner extends EpisodicOfflineLearner[Int](2.0, 0.95, schedule, batchSize = 4):
    var gradientSteps: Int = 0
    var sessions: Int = 0
    def recorded: Int = transitions.size
    override def policy: AgentState => Double = _ => 1.0
    override protected def record(trajectory: Trajectory): Unit = trajectory.foreach(_ => transitions.add(1, true))
    override protected def trainOn(batch: Vector[Int], manager: NDManager): Unit = gradientSteps += 1
    override protected def onTrained(): Unit = sessions += 1

  "An episodic offline learner" should "train after each exploration episode and once more, longer, at the end" in {
    val learner = CountingLearner()
    (1 to 5).foreach(_ => learner.endEpisode(Seq(trajectory)))
    learner.sessions shouldBe 3
    learner.gradientSteps shouldBe 3 + 3 + 3 * 2
  }

  it should "stop storing transitions once it no longer trains" in {
    val learner = CountingLearner()
    (1 to 5).foreach(_ => learner.endEpisode(Seq(trajectory, trajectory)))
    learner.recorded shouldBe 3 * 2
  }

  it should "not train without transitions" in {
    val learner = CountingLearner()
    learner.endEpisode(Seq.empty)
    learner.sessions shouldBe 0
  }
end EpisodicOfflineLearnerSpec
