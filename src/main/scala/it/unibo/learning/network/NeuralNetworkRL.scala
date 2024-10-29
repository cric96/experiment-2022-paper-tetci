package it.unibo.learning.network

import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.torch.torch
import it.unibo.util.TemporalInfo
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.SeqConverters

object NeuralNetworkRL {

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
