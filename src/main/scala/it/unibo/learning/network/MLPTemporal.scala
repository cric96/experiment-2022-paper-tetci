package it.unibo.learning.network
import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.NeuralNetworkRL.Historical
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.SeqConverters

class MLPTemporal(snapshots: Int, hiddenSize: Int, val actionSpace: List[Double]) extends NeuralNetworkRL {
  override val underlying: py.Dynamic = DQN(snapshots, hiddenSize, actionSpace.size)
  override def encode(state: AgentState): py.Any = Historical.encodeHistory(state, snapshots)

  override def encodeBatch(seq: Seq[py.Any], device: py.Any): py.Dynamic =
    normalize(torch.tensor(seq.toPythonCopy, device = device))

  override def policy(device: py.Any): (AgentState) => (Int, Contextual) =
    NeuralNetworkRL.policyFromNetwork(this, Seq(1, snapshots), device)

  override def cloneNetwork: NeuralNetworkRL = new MLPTemporal(snapshots, hiddenSize, actionSpace)

  override def emptyContextual: Contextual = ()

  override def normalize(input: py.Dynamic): py.Dynamic = {
    val result = torch.nn.functional.normalize(input)
    input.del()
    result
  }
}
