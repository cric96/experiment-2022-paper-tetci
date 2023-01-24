package it.unibo.learning.network
import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

class MLPTemporal(snapshots: Int, hiddenSize: Int, val actionSpace: List[Double]) extends NeuralNetworkRL {
  override val underlying: py.Dynamic = DQN(snapshots, hiddenSize, actionSpace.size)
  override def encode(state: AgentState): py.Any = {
    val states: LazyList[Double] = state.neighborhoodOutput
      .map(_(state.me))
      .map(_.data)
      .map {
        case x if !x.isFinite => -1
        case x => x
      }
      .to(LazyList)
    val fill: LazyList[Double] = LazyList.continually(0.0)
    (states #::: fill).take(snapshots).toPythonCopy
  }

  override def encodeBatch(seq: Seq[py.Any], device: py.Any): py.Dynamic =
    torch.tensor(seq.toPythonCopy, device = device)

  override def policy(device: py.Any): (AgentState) => (Int, Contextual) = state => {
    val netInput = encode(state)
    val session = PythonMemoryManager.session()
    // context
    import session._
    py.`with`(torch.no_grad()) { _ =>
      val tensor = torch.tensor(netInput).record().view(1, snapshots).record().to(device).record()
      val netOutput = underlying(tensor).record()
      val elements = netOutput.tolist().record().bracketAccess(0).record()
      val max = py.Dynamic.global.max(elements)
      val index = elements.index(max).as[Int]
      session.clear()
      (index, ())
    }
  }

  override def cloneNetwork: NeuralNetworkRL = new MLPTemporal(snapshots, hiddenSize, actionSpace)

  override def emptyContextual: Contextual = ()
}
