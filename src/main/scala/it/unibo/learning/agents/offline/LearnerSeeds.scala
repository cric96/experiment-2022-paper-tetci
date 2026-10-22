package it.unibo.learning.agents.offline

import scala.util.Random

/** Seeds of the offline learners, derived from the seed of the run and from the trade-off weight, so that every (run,
  * β) pair gets distinct but reproducible randomness and network initializations.
  */
object LearnerSeeds {

  /** Multiplier that spreads the run seed before mixing it with the weight, for the source of randomness. */
  val RandomMultiplier: Long = 31L

  /** Multiplier that spreads the run seed before mixing it with the weight, for the network initializations. */
  val NetworkMultiplier: Int = 1009

  def random(runSeed: Double, weight: Double): Random =
    new Random((runSeed.toLong * RandomMultiplier) ^ java.lang.Double.doubleToLongBits(weight))

  /** Base seed of the networks of a learner; each network adds its own offset. */
  def networks(runSeed: Double, weight: Double): Int = (runSeed.toInt * NetworkMultiplier) ^ weight.hashCode()
}
