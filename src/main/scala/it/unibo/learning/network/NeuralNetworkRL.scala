package it.unibo.learning.network

import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.torch.{PythonMemoryManager, torch}
import it.unibo.util.TemporalInfo
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.SeqConverters

/** An NN used in the context of RL */
trait NeuralNetworkRL {
  val underlying: py.Dynamic
  def forward(input: py.Dynamic)(implicit session: PythonMemoryManager.Session): py.Dynamic = underlying(input)
  def actionSpace: List[Double]
  def emptyContextual: Contextual
  def cloneNetwork: NeuralNetworkRL
  def encode(state: AgentState): py.Any
  def encodeBatch(seq: Seq[py.Any], device: py.Any)(implicit session: PythonMemoryManager.Session): py.Dynamic
  def policy(device: py.Any): (AgentState) => (Int, Contextual)
  def normalize(input: py.Dynamic): py.Dynamic
}

object NeuralNetworkRL {

  def policyFromNetwork(nn: NeuralNetworkRL, inputShape: Seq[Int], device: py.Any): (AgentState) => (Int, Contextual) =
    state => {
      val netInput = nn.encode(state)
      val session = PythonMemoryManager.session()
      // context
      import session._
      py.`with`(torch.no_grad()) { _ =>
        val tensor = torch
          .tensor(netInput)
          .record()
          .applyDynamic("view")(inputShape.map(_.as[py.Any]): _*)
          .record()
          .to(device)
        val normalized = nn.normalize(tensor).record()
        val netOutput = nn.underlying(normalized).record()
        val elements = netOutput.tolist().record().bracketAccess(0).record()
        val max = py.Dynamic.global.max(elements)
        val index = elements.index(max).as[Int]
        session.clear()
        (index, ())
      }
    }

  object Historical {
    def encodeHistory(state: AgentState, snapshots: Int): py.Any = {
      // TemporalInfo.computeDeltaTrend(me.map(_.data))
      val states: LazyList[Double] = state.neighborhoodOutput
        .map(_(state.me))
        .map(_.data)
        .replaceInfinite()
        .reverse
        .to(LazyList)
      val fill: LazyList[Double] = LazyList.continually(0.0)
      (states #::: fill).take(snapshots).toPythonCopy
    }
  }
  object Spatial {
    def encodeSpatialUnbounded(state: AgentState, considerAction: Boolean): py.Any = {
      val currentSnapshot = state.neighborhoodOutput.head.toList.sortBy(_._2.distance)
      val data = currentSnapshot.map(_._2.data).replaceInfinite() to LazyList
      if (considerAction) {
        val actions = currentSnapshot.map(_._2.oldAction)
        data.zip(actions).map { case (data, action) => List(data, action.toDouble).toPythonCopy }.toPythonCopy
      } else {
        data.toPythonCopy
      }
    }
    def encodeSpatial(state: AgentState, neigh: Int, considerAction: Boolean): py.Any = {
      val states: LazyList[Double] = {
        val currentSnapshot = state.neighborhoodOutput.head.toList.sortBy(_._2.distance).take(neigh)
        val data = currentSnapshot.map(_._2.data).replaceInfinite() to LazyList
        if (considerAction) {
          val actions = currentSnapshot.map(_._2.oldAction)
          data.zip(actions).flatMap { case (data, action) => List(data, action.toDouble) }
        } else {
          data
        }
      }
      val fill: LazyList[Double] = LazyList.continually(0.0)
      (states #::: fill).take(neigh * (if (considerAction) 2 else 1)).toPythonCopy
    }
  }
}
