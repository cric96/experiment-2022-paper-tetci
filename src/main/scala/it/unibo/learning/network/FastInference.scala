package it.unibo.learning.network

import ai.djl.nn.Block
import ai.djl.nn.core.Linear

import scala.jdk.CollectionConverters._

/** Forward passes re-implemented on a snapshot of the network parameters. Policies are evaluated by every node in every
  * round: going through DJL for single samples is slow and serialized, so the nodes use these lock-free
  * single-precision versions, refreshed after each training session.
  */
object FastInference {

  /** A dense layer y = W·x + b, with W stored row-major as (outputs × inputs). */
  final case class DenseLayer(weights: Array[Float], bias: Array[Float], inputs: Int, outputs: Int) {

    /** Hot path: plain loops, accumulating in the same order as the original implementation. */
    def apply(input: Array[Float], relu: Boolean): Array[Float] = {
      val output = new Array[Float](outputs)
      var unit = 0
      while (unit < outputs) {
        var sum = bias(unit)
        val row = unit * inputs
        var feature = 0
        while (feature < inputs) {
          sum += weights(row + feature) * input(feature)
          feature += 1
        }
        output(unit) = if (relu && sum < 0.0f) 0.0f else sum
        unit += 1
      }
      output
    }
  }

  object DenseLayer {

    /** Copies the current parameters of a DJL linear layer. */
    def of(linear: Block): DenseLayer = {
      val parameters = linear.getParameters
      val weights = parameters.get("weight").getArray
      val outputs = weights.getShape.get(0).toInt
      val bias = Option(parameters.get("bias")).fold(new Array[Float](outputs))(_.getArray.toFloatArray)
      DenseLayer(weights.toFloatArray, bias, weights.getShape.get(1).toInt, outputs)
    }
  }

  /** The linear layers of `block`, in order. */
  def denseLayersOf(block: Block): Seq[DenseLayer] =
    block.getChildren.values().asScala.toSeq.collect { case linear: Linear => DenseLayer.of(linear) }

  /** A multi-layer perceptron with ReLU after every layer but the last, optionally followed by tanh. */
  def mlp(layers: Seq[DenseLayer], tanhOutput: Boolean): Array[Float] => Array[Float] = { input =>
    val hidden = layers.init.foldLeft(input)((activation, layer) => layer(activation, relu = true))
    val output = layers.last(hidden, relu = false)
    if (tanhOutput) output.map(value => math.tanh(value.toDouble).toFloat) else output
  }

  /** Numerically stable softmax, in single precision. */
  def softmax(logits: Array[Float]): Array[Float] = {
    val maxLogit = logits.foldLeft(Float.NegativeInfinity)(math.max)
    val exponentials = logits.map(logit => math.exp((logit - maxLogit).toDouble).toFloat)
    val total = exponentials.sum
    exponentials.map(_ / total)
  }
}
