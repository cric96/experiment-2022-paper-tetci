package it.unibo.learning.network

import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.NeuralNetworkRL.Spatial
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

class GNNSpatial(hiddenSize: Int, val actionSpace: List[Double], considerAction: Boolean = false)
    extends NeuralNetworkRL {
  val True = torch.tensor(Seq(true).toPythonCopy)

  val dataSpaceMultiplier = if (considerAction) 2 else 1

  override val underlying: py.Dynamic = GNNDQN(dataSpaceMultiplier, hiddenSize, actionSpace.size)

  override def forward(input: py.Dynamic): py.Dynamic = {
    val in = input.bracketAccess(0)
    val mask = input.bracketAccess(1)
    underlying(normalize(in.x), in.edge_index).bracketAccess(mask)
  }

  override def encode(state: AgentState): py.Any = {
    val x = torch.tensor(Spatial.encodeSpatialUnbounded(state, considerAction))
    val index = computeNeighborhoodIndex(py.Dynamic.global.len(x).as[Int] - 1)
    geometric.data.Data(x = x, edge_index = index)
  }

  override def encodeBatch(seq: Seq[py.Any], device: py.Any): py.Dynamic = {
    val data = geometric.data.Batch.from_data_list(seq.toPythonCopy)
    val neigh = seq.map(_.as[py.Dynamic]).map(_.x).map(_.shape.bracketAccess(0) - 1)
    val masks = neigh.map(neighCount => torch.zeros(neighCount, dtype = torch.bool))
    val maskWithMe = masks.map(mask => torch.cat(Seq(True, mask).toPythonCopy))
    val flattenMask = torch.cat(maskWithMe.toPythonCopy)
    py"$data, $flattenMask"
  }

  override def policy(device: py.Any): (AgentState) => (Int, Contextual) =
    // NeuralNetworkRL.policyFromNetwork(this, Seq(1, neigh * dataSpaceMultiplier), device)
    state => {
      val netInput = encode(state).as[py.Dynamic]
      py.`with`(torch.no_grad()) { _ =>
        val data = underlying(normalize(netInput.x), netInput.edge_index) // .bracketAccess(0)
        val actionIndex = data.bracketAccess(0).max(0).bracketAccess(1).item().as[Int]
        (actionIndex, ())
      // actionSpace.head
      }
    }
  override def cloneNetwork: NeuralNetworkRL = new GNNSpatial(hiddenSize, actionSpace, considerAction)

  override def emptyContextual: Contextual = ()

  override def normalize(input: py.Dynamic): py.Dynamic = {
    val result = torch.nn.functional.normalize(input)
    input.del()
    result
  }

  def computeNeighborhoodIndex(num: Int): py.Dynamic = {
    val neighborhoodIndex =
      List(
        List(0) ::: List.fill(num)(0) ::: List.range(1, num + 1),
        List(0) ::: (List.range(1, num + 1)) ::: List.fill(num)(0)
      )
    val neighborhoodIndexPython = neighborhoodIndex.map(_.toPythonCopy).toPythonCopy
    torch.tensor(neighborhoodIndexPython)
  }
}
