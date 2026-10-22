package it.unibo.learning

/** Configuration of the learning loop: the seed of the run, the capacity of the shared replay buffer, the size of the
  * online training batches, and the length of an episode (in steps of the central agent, one per time unit).
  */
final case class LearningInfo(randomSeed: Int, bufferSize: Int, batchSize: Int, episodeSize: Int)
