package it.unibo.learning

/** The arena of the deployments: a square of side `width` with a node every `step`, each moved randomly by up to
  * `randomness` along each axis.
  */
final case class Box(width: Double, step: Double, randomness: Double)
