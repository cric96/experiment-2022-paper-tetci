package it.unibo.learning.network

import ai.djl.ndarray.NDArray

/** Arithmetic operators on DJL arrays, so that losses and targets read like their formulas. */
object NDArrayOps {
  extension (self: NDArray) {
    def +(other: NDArray): NDArray = self.add(other)
    def +(scalar: Number): NDArray = self.add(scalar)
    def -(other: NDArray): NDArray = self.sub(other)
    def -(scalar: Number): NDArray = self.sub(scalar)
    def *(other: NDArray): NDArray = self.mul(other)
    def *(scalar: Number): NDArray = self.mul(scalar)
    def /(other: NDArray): NDArray = self.div(other)
    def /(scalar: Number): NDArray = self.div(scalar)
    def unary_- : NDArray = self.neg()
  }
}
