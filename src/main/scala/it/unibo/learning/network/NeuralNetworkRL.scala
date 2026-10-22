package it.unibo.learning.network

import it.unibo.learning.abstractions.AgentState
import it.unibo.util.TemporalInfo

object NeuralNetworkRL {

  object Historical {
    def encodeHistory(state: AgentState, snapshots: Int): Array[Double] = {
      val outputs = state.neighborhoodOutput
        .flatMap(_.get(state.me))
        .map(_.data)
        .replaceInfinite()
        .takeRight(snapshots + 1)
      val trends = TemporalInfo
        .computeDeltaTrend(outputs)
        .reverse
        .to(LazyList)
      val fill: LazyList[Double] = LazyList.continually(0.0)
      (trends #::: fill).take(snapshots).toArray
    }
  }

  /** Rich 9D Spatial-Temporal + Topological Role + Weight-Conditioned encoder: 0: latest self output delta trend in
    * {-1, 0, +1} 1: previous self output delta trend in {-1, 0, +1} 2: self output change magnitude tanh(|g_i(t) -
    * g_i(t-1)| / 50) in [0, 1] 3: max neighbor output change magnitude tanh(max_{j!=i} |g_j(t) - g_j(t-1)| / 50) in [0,
    * 1] 4: fraction of 1-hop neighbors whose output changed in the last step in [0, 1] 5: local Bellman-Ford residual
    * tanh(|g_i(t) - min_{j!=i}(g_j(t) + d_ij)| / 50) in [0, 1] 6: downstream children ratio in shortest-path tree in
    * [0, 1] (topological backbone vs leaf role) 7: normalized minimum wake-up period among 1-hop neighbors in [0, 1] 8:
    * convergence trade-off weight beta in [0, 1]
    */
  object SpatialTemporal {
    val FeatureDim: Int = 9

    /** Fraction of 1-hop neighbors that route their shortest path through this node (g_j approx g_i + d_ij). */
    def downstreamChildrenRatio(state: AgentState): Double = {
      state.neighborhoodOutput.lastOption match {
        case Some(curr) =>
          val myValOpt = curr.get(state.me).map(_.data)
          val others = curr.filter(_._1 != state.me)
          myValOpt match {
            case Some(myVal) if myVal.isFinite && others.nonEmpty =>
              val children = others.count { case (_, info) =>
                info.data.isFinite && info.data > myVal &&
                math.abs(info.data - (myVal + info.distance)) <= math.max(8.0, 0.2 * info.distance)
              }
              children.toDouble / others.size.toDouble
            case _ => 0.0
          }
        case None => 0.0
      }
    }

    def localScale(state: AgentState): Double = {
      state.neighborhoodOutput.lastOption match {
        case Some(curr) =>
          val finiteVals = curr.values.map(_.data).filter(v => v.isFinite && math.abs(v) < 1e6)
          if (finiteVals.isEmpty) 50.0
          else math.max(1.0, 0.2 * finiteVals.map(math.abs).max)
        case None => 50.0
      }
    }

    /** Continuous instability intensity I(s) in [0, 1] combining self history, 1-hop neighborhood, and topological
      * role.
      */
    def instabilityIntensity(state: AgentState): Double = {
      val scale = localScale(state)
      val myOutputs = state.neighborhoodOutput.flatMap(_.get(state.me)).takeRight(4).map(_.data)
      val selfTrend = TemporalInfo.computeDeltaTrend(myOutputs)
      val latestSelfChanged = selfTrend.takeRight(2).exists(_ != 0.0)
      val cooldownSelfChanged = selfTrend.exists(_ != 0.0)

      val snaps = state.neighborhoodOutput.takeRight(2)
      val (maxNeighDelta, neighRatio) = if (snaps.size == 2) {
        val prev = snaps.head
        val curr = snaps.last
        val others = curr.filter(_._1 != state.me)
        if (others.isEmpty) (0.0, 0.0)
        else {
          var maxD = 0.0
          var changed = 0
          others.foreach { case (id, info) =>
            prev.get(id).foreach { p =>
              if (p.data != info.data) {
                changed += 1
                val d =
                  if (p.data.isFinite && info.data.isFinite) math.abs(info.data - p.data)
                  else scale * 2.0
                if (d > maxD) maxD = d
              }
            }
          }
          (math.tanh(maxD / scale), changed.toDouble / others.size.toDouble)
        }
      } else (0.0, 0.0)

      val bfResidual = bellmanFordResidual(state)
      val childrenRatio = downstreamChildrenRatio(state)
      val roleMultiplier = 0.75 + 0.5 * childrenRatio

      val rawIntensity = if (latestSelfChanged) {
        1.0
      } else if (maxNeighDelta > 1e-3 || neighRatio > 0.0 || bfResidual > 0.05) {
        math.min(1.0, math.max(0.45, math.max(maxNeighDelta, math.max(neighRatio, bfResidual))) * roleMultiplier)
      } else if (cooldownSelfChanged) {
        0.35 * roleMultiplier
      } else {
        0.0
      }
      math.min(1.0, rawIntensity)
    }

    def bellmanFordResidual(state: AgentState): Double = {
      val scale = localScale(state)
      val snaps = state.neighborhoodOutput.takeRight(2)
      if (snaps.size == 2) {
        val prev = snaps.head
        val curr = snaps.last
        val myCurr = curr.get(state.me).map(_.data).getOrElse(0.0)
        val myPrev = prev.get(state.me).map(_.data).getOrElse(0.0)
        val others = curr.filter(_._1 != state.me)
        if (others.isEmpty || !myCurr.isFinite || !myPrev.isFinite) 0.0
        else {
          val maxEdgeDrift = others.foldLeft(0.0) { case (acc, (id, info)) =>
            prev.get(id) match {
              case Some(pInfo) if info.data.isFinite && pInfo.data.isFinite =>
                val diffNow = myCurr - info.data
                val diffPrev = myPrev - pInfo.data
                math.max(acc, math.abs(diffNow - diffPrev))
              case _ => acc
            }
          }
          math.tanh(maxEdgeDrift / scale)
        }
      } else 0.0
    }

    def encode(state: AgentState, weight: Double, maxAction: Double = 5.0): Array[Float] = {
      val scale = localScale(state)
      val trends = Historical.encodeHistory(state, 2)
      val deltaLatest = trends(0).toFloat
      val deltaPrev = trends(1).toFloat

      val selfHistory = state.neighborhoodOutput.flatMap(_.get(state.me)).takeRight(2).map(_.data).replaceInfinite()
      val selfDeltaMag = if (selfHistory.size >= 2) {
        math.tanh(math.abs(selfHistory.last - selfHistory.head) / scale).toFloat
      } else 0.0f

      val snaps = state.neighborhoodOutput.takeRight(2)
      val (maxNeighDeltaMag, neighChangedRatio) = if (snaps.size == 2) {
        val prev = snaps.head
        val curr = snaps.last
        val others = curr.filter(_._1 != state.me)
        if (others.isEmpty) (0.0f, 0.0f)
        else {
          var maxD = 0.0
          var changedCount = 0
          others.foreach { case (id, info) =>
            prev.get(id).foreach { p =>
              if (p.data != info.data) {
                changedCount += 1
                val d =
                  if (p.data.isFinite && info.data.isFinite) math.abs(info.data - p.data)
                  else scale * 2.0
                if (d > maxD) maxD = d
              }
            }
          }
          (math.tanh(maxD / scale).toFloat, changedCount.toFloat / others.size.toFloat)
        }
      } else (0.0f, 0.0f)

      val bfRes = bellmanFordResidual(state).toFloat
      val childrenRatio = downstreamChildrenRatio(state).toFloat

      val minNeighAction = state.neighborhoodOutput.lastOption
        .flatMap { snap =>
          val otherActions = snap.collect { case (id, info) if id != state.me => info.oldAction }
          otherActions.minOption
        }
        .map(a => (a / maxAction).toFloat)
        .getOrElse(1.0f)

      val energyW = math.max(0.005, 1.0 - weight)
      // Maps weight in [0.5, 1.0] smoothly to [-1.0, +1.0] according to the steady-state trade-off ratio
      val betaFeature = (2.0 * (energyW / (energyW + 0.06 * weight)) - 1.0).toFloat

      Array(
        deltaLatest, deltaPrev, selfDeltaMag, maxNeighDeltaMag, neighChangedRatio, bfRes, childrenRatio, minNeighAction,
        betaFeature
      )
    }
  }

  /** Star-Graph encoder for EdgeGAT (Graph Attention + Relational Max-Pooling GNN): Encodes the ego node (5 features) +
    * up to MaxNeighbors = 8 1-hop neighbors (6 node/edge features each).
    */
  object StarGraph {
    val SelfDim: Int = 5
    val MaxNeighbors: Int = 8
    val NeighDim: Int = 6
    val TotalDim: Int = SelfDim + MaxNeighbors * NeighDim

    def encode(state: AgentState, weight: Double, maxAction: Double = 5.0): Array[Float] = {
      val scale = SpatialTemporal.localScale(state)
      val trends = Historical.encodeHistory(state, 2)
      val deltaLatest = trends(0).toFloat
      val deltaPrev = trends(1).toFloat

      val selfHistory = state.neighborhoodOutput.flatMap(_.get(state.me)).takeRight(2).map(_.data).replaceInfinite()
      val selfDeltaMag = if (selfHistory.size >= 2) {
        math.tanh(math.abs(selfHistory.last - selfHistory.head) / scale).toFloat
      } else 0.0f

      val childrenRatio = SpatialTemporal.downstreamChildrenRatio(state).toFloat
      val energyW = math.max(0.005, 1.0 - weight)
      val betaFeature = (2.0 * (energyW / (energyW + 0.06 * weight)) - 1.0).toFloat

      val out = new Array[Float](TotalDim)
      out(0) = deltaLatest
      out(1) = deltaPrev
      out(2) = selfDeltaMag
      out(3) = childrenRatio
      out(4) = betaFeature

      val snaps = state.neighborhoodOutput.takeRight(2)
      val currSnap = snaps.lastOption.getOrElse(Map.empty)
      val prevSnap = if (snaps.size == 2) snaps.head else currSnap
      val myCurr = currSnap.get(state.me).map(_.data).getOrElse(0.0)
      val myPrev = prevSnap.get(state.me).map(_.data).getOrElse(0.0)

      val sortedNeighbors = currSnap
        .filter(_._1 != state.me)
        .toList
        .sortBy(_._2.distance)
        .take(MaxNeighbors)

      var k = 0
      while (k < MaxNeighbors) {
        val baseOffset = SelfDim + k * NeighDim
        if (k < sortedNeighbors.size) {
          val (nId, info) = sortedNeighbors(k)
          val prevInfoOpt = prevSnap.get(nId)
          val nCurr = if (info.data.isFinite) info.data else scale * 2.0
          val nPrev = prevInfoOpt.map(p => if (p.data.isFinite) p.data else scale * 2.0).getOrElse(nCurr)
          val diff = nCurr - nPrev
          val nSign = math.signum(diff).toFloat
          val nMag = math.tanh(math.abs(diff) / scale).toFloat
          val edgeDrift = if (myCurr.isFinite && myPrev.isFinite) {
            math.tanh(math.abs((myCurr - nCurr) - (myPrev - nPrev)) / scale).toFloat
          } else 0.0f
          val relOrder = if (myCurr.isFinite) {
            math.tanh((nCurr - myCurr) / scale).toFloat
          } else 0.0f
          val normDist = math.min(1.0, info.distance / 100.0).toFloat
          val normClock = (info.oldAction / maxAction).toFloat

          out(baseOffset + 0) = nSign
          out(baseOffset + 1) = nMag
          out(baseOffset + 2) = edgeDrift
          out(baseOffset + 3) = relOrder
          out(baseOffset + 4) = normDist
          out(baseOffset + 5) = normClock
        } else {
          // Idle padding neighbor slot (zero delta, max clock = 1.0)
          out(baseOffset + 5) = 1.0f
        }
        k += 1
      }
      out
    }
  }

}
