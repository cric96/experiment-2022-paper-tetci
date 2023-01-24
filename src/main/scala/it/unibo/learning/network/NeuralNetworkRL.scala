package it.unibo.learning.network

import it.unibo.learning.abstractions.{AgentState, Contextual}
import me.shadaj.scalapy.py

/** An NN used in the context of RL */
trait NeuralNetworkRL {
  val underlying: py.Dynamic
  def actionSpace: List[Double]
  def emptyContextual: Contextual
  def cloneNetwork: NeuralNetworkRL
  def encode(state: AgentState): py.Any
  def encodeBatch(seq: Seq[py.Any], device: py.Any): py.Dynamic
  def policy(device: py.Any): (AgentState) => (Int, Contextual)
}
