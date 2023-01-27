package it.unibo.learning.network

import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.NeuralNetworkRL.Spatial
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

class MLPSpatial(neigh: Int, hiddenSize: Int, val actionSpace: List[Double], considerAction: Boolean = false)
    extends NeuralNetworkRL {
  val dataSpaceMultiplier = if (considerAction) 2 else 1
  override val underlying: py.Dynamic = DQN(neigh * dataSpaceMultiplier, hiddenSize, actionSpace.size)

  override def encode(state: AgentState): py.Any = Spatial.encodeSpatial(state, neigh, considerAction)

  override def encodeBatch(seq: Seq[py.Any], device: py.Any)(implicit
      session: PythonMemoryManager.Session
  ): py.Dynamic =
    normalize(torch.tensor(seq.toPythonCopy, device = device))

  override def policy(device: py.Any): (AgentState) => (Int, Contextual) =
    NeuralNetworkRL.policyFromNetwork(this, Seq(1, neigh * dataSpaceMultiplier), device)

  override def cloneNetwork: NeuralNetworkRL = new MLPSpatial(neigh, hiddenSize, actionSpace, considerAction)

  override def emptyContextual: Contextual = ()

  override def normalize(input: py.Dynamic): py.Dynamic = {
    val result = torch.nn.functional.normalize(input)
    input.del()
    result
  }
}
