package it.unibo.learning.network

import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.NeuralNetworkRL.Spatial
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

class GNNSpatial(hiddenSize: Int, val actionSpace: List[Double], considerAction: Boolean = false)
    extends NeuralNetworkRL {
  val True = torch.tensor(Seq(true).toPythonCopy)

  val dataSpaceMultiplier = if (considerAction) 2 else 1

  override val underlying: py.Dynamic = GNNDQN(dataSpaceMultiplier, hiddenSize, actionSpace.size)

  override def forward(input: py.Dynamic)(implicit session: PythonMemoryManager.Session): py.Dynamic = {
    import session._
    val converted = input.as[Seq[py.Dynamic]].map(_.record())
    underlying(normalize(converted(0)).record(), converted(1)).record().bracketAccess(converted(2).record())
  }

  override def encode(state: AgentState): py.Any = {
    val x = Spatial.encodeSpatialUnbounded(state, considerAction)
    val index = computeNeighborhoodIndex(py.Dynamic.global.len(x).as[Int] - 1)
    (x, index)
  }
  override def encodeBatch(seq: Seq[py.Any], device: py.Any)(implicit
      session: PythonMemoryManager.Session
  ): py.Dynamic = {
    import session._
    val dataRaw = scala.collection.mutable.Buffer.empty[py.Dynamic]
    val indexRawStart = scala.collection.mutable.Buffer.empty[py.Any]
    val indexRawEnd = scala.collection.mutable.Buffer.empty[py.Any]
    val maskRaw = scala.collection.mutable.Buffer.empty[py.Any]
    var max = 0
    val unsafe = seq.map(_.as[py.Dynamic])
    for (elem <- unsafe) {
      val x = elem.bracketAccess(0).as[Seq[py.Dynamic]]
      val y = elem.bracketAccess(1).as[Seq[py.Dynamic]].map(_.as[Seq[Int]])
      val yUpdated = y.map(links => links.map(_ + max))
      max = x.size + max
      dataRaw.addAll(x)
      indexRawStart.addAll(yUpdated.head.map(_.as[py.Any]))
      indexRawEnd.addAll(yUpdated.head.map(_.as[py.Any]))
      maskRaw.addAll(true :: List.fill(x.size - 1)(false).map(_.as[py.Any]))
    }
    val x = torch.tensor(dataRaw.toPythonCopy, device = device).record()
    val index =
      torch.tensor(Seq(indexRawStart.toPythonCopy, indexRawEnd.toPythonCopy).toPythonCopy, device = device).record()
    val mask = torch.tensor(maskRaw.toPythonCopy, device = device).record()
    Seq(x, index, mask).toPythonCopy.as[py.Dynamic]
  }

  override def policy(device: py.Any): (AgentState) => (Int, Contextual) =
    // NeuralNetworkRL.policyFromNetwork(this, Seq(1, neigh * dataSpaceMultiplier), device)
    state => {
      val session = PythonMemoryManager.session()
      // context
      import session._
      val unsafeState = encode(state).as[py.Dynamic]
      val x = torch.tensor(unsafeState.bracketAccess(0).record()).record()
      val edge_index = torch.tensor(unsafeState.bracketAccess(1).record()).record()

      py.`with`(torch.no_grad()) { _ =>
        val data = underlying(normalize(x.to(device)).record(), edge_index.to(device).record()).record()
        val elements = data.tolist().record().bracketAccess(0).record()
        val max = py.Dynamic.global.max(elements)
        val index = elements.index(max).as[Int]
        session.clear()
        (index, ())
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
    neighborhoodIndexPython.as[py.Dynamic]
  }
}
