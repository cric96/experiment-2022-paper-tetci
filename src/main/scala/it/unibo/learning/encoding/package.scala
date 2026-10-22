package it.unibo.learning

import scala.collection.SeqOps

/** Encoders that turn what a node observed ([[abstractions.AgentState]]) into features for the policies. */
package object encoding {
  implicit final class InfiniteValuesOps[F[A] <: SeqOps[A, F, F[A]]](private val values: F[Double]) extends AnyVal {

    /** Replaces the non-finite values (e.g. the gradient of a node with no source) with −1. */
    def replaceInfinite(): F[Double] = values.map(value => if value.isFinite then value else -1.0)
  }
}
