package it.unibo.learning

import scala.collection.SeqOps

package object network {
  implicit class NormalizeIterable[F[A] <: SeqOps[A, F, F[A]]](f: F[Double]) {
    def replaceInfinite(): F[Double] = f.map {
      case x if x.isInfinite => -1
      case x => x
    }
  }
}
