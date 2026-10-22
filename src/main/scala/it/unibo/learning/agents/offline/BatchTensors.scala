package it.unibo.learning.agents.offline

import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.{NDArray, NDManager}

/** Builders of the tensors of a training batch. */
object BatchTensors {

  /** One row per sample. */
  def rows(values: Seq[Array[Float]], manager: NDManager): NDArray = manager.create(values.toArray)

  /** A (samples × 1) column. */
  def column(values: Seq[Float], manager: NDManager): NDArray =
    manager.create(values.toArray, new Shape(values.size.toLong, 1))

  /** A (samples × 1) column of indices. */
  def indexColumn(values: Seq[Int], manager: NDManager): NDArray =
    manager.create(values.map(_.toLong).toArray, new Shape(values.size.toLong, 1))
}
