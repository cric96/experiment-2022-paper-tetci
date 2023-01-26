package it.unibo.learning.network

import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.NeuralNetworkRL.Historical
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

class RNNRL(snapshots: Int, hiddenSize: Int, val actionSpace: List[Double]) extends NeuralNetworkRL {
  override val underlying: py.Dynamic = RDQN(1, hiddenSize, actionSpace.size, snapshots)
  override def encode(state: AgentState): py.Any = Historical.encodeHistory(state, snapshots)

  override def encodeBatch(seq: Seq[py.Any], device: py.Any): py.Dynamic = {
    val base = torch.tensor(seq.toPythonCopy, device = device)
    val reshaped = normalize(base.view((seq.size, snapshots, 1)))
    base.del()
    reshaped
  }

  override def policy(device: py.Any): (AgentState) => (Int, Contextual) =
    NeuralNetworkRL.policyFromNetwork(this, Seq(1, snapshots, 1), device)

  override def cloneNetwork: NeuralNetworkRL = new RNNRL(snapshots, hiddenSize, actionSpace)

  override def emptyContextual: Contextual = ()

  override def normalize(input: py.Dynamic): py.Dynamic = {
    val result = torch.nn.functional.normalize(input, dim = 1)
    input.del()
    result
  }

}
