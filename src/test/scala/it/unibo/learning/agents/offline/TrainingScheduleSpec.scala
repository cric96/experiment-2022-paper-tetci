package it.unibo.learning.agents.offline

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TrainingScheduleSpec extends AnyFlatSpec with Matchers {
  private val schedule = TrainingSchedule(explorationEpisodes = 10, stepsPerEpisode = 100, finalTrainingMultiplier = 3)

  "Epsilon" should "decay linearly from its initial value to zero during exploration" in {
    schedule.epsilonAt(0) shouldBe schedule.initialEpsilon
    schedule.epsilonAt(5) shouldBe (schedule.initialEpsilon / 2) +- 1e-12
    schedule.epsilonAt(10) shouldBe 0.0
    schedule.epsilonAt(15) shouldBe 0.0
  }

  "The schedule" should "explore only during the exploration episodes" in {
    (0 to 11).map(schedule.isExploring) shouldBe Seq.fill(10)(true) ++ Seq(false, false)
  }

  it should "train after each exploration episode, longer after the last one, then never again" in {
    schedule.trainingStepsAfter(0) shouldBe Some(100)
    schedule.trainingStepsAfter(9) shouldBe Some(100)
    schedule.trainingStepsAfter(10) shouldBe Some(300)
    schedule.trainingStepsAfter(11) shouldBe None
  }
}
