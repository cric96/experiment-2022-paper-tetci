package it.unibo.testing

import org.scalatest.Tag

/** Tests that pin exact values produced by the PyTorch CPU kernels (training, initialization, inference): they guard
  * refactorings on the development machine, but other CPUs may legitimately produce different bits, so the CI excludes
  * them.
  */
object PyTorchNumerics extends Tag("it.unibo.testing.PyTorchNumerics")
