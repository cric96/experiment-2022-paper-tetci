package it.unibo.learning.encoding

import scala.collection.SeqOps

extension [F[A] <: SeqOps[A, F, F[A]]](values: F[Double]) {

  /** Replaces the non-finite values (e.g. the gradient of a node with no source) with −1. */
  def replaceInfinite(): F[Double] = values.map(value => if value.isFinite then value else -1.0)
}
