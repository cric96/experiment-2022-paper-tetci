package it.unibo.alchemist.model.implementations.timedistributions.reactions

import it.unibo.alchemist.loader.deployments.Grid
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces._
import it.unibo.learning.{Box, LearningInfo}
import it.unibo.learning.abstractions.{AgentState, ReplayBuffer, Step}
import it.unibo.learning.agents.Learner
import it.unibo.scafi.Sensors
import org.apache.commons.math3.random.RandomGenerator

import scala.collection.immutable.Queue
import scala.jdk.CollectionConverters.{IteratorHasAsScala, MapHasAsScala}
import scala.util.Random

class CentralAgent[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    random: RandomGenerator,
    learningInfo: LearningInfo,
    actionSpace: List[Double],
    learner: Learner,
    environmentBox: Box
) extends AbstractGlobalReaction[T, P](environment, distribution) {
  private val scalaRandom = new Random(learningInfo.randomSeed)
  private var stepTicks = 0
  private var episodeCounter = 0
  private var cachedSourceKey: (Int, Set[Int]) = (-1, Set.empty)
  private var cachedGroundTruth: Map[Int, Double] = Map.empty
  private var cachedHopTruth: Map[Int, Double] = Map.empty
  private var cachedCollectTruth: Map[Int, Double] = Map.empty
  private var cachedBroadcastTruth: Double = 0.0
  private val spatialRows = scala.collection.mutable.ArrayBuffer.empty[String]
  private val spatialEdges = scala.collection.mutable.ArrayBuffer.empty[String]
  private val gnnSpatialRows = scala.collection.mutable.ArrayBuffer.empty[String]
  // used to restart the simulation with the same configuration
  private var initialSnapshot: List[Node[T]] =
    List.empty[Node[T]]
  val memory: ReplayBuffer = new ReplayBuffer(learningInfo.bufferSize, scalaRandom)
  learner.injectRandom(scalaRandom)

  private val evalBetaSchedule = Array(0.50, 0.80, 0.95, 0.99, 1.00)

  def policy = {
    if (episodeCounter >= 10 && managers.headOption.exists(_.has("program"))) {
      val idx = math.min(evalBetaSchedule.length - 1, (episodeCounter - 10) / 2)
      learner.policyForWeight(evalBetaSchedule(idx))
    } else {
      learner.policy
    }
  }

  override def executeBeforeUpdateDistribution(): Unit = {
    val currentTime = environment.getSimulation.getTime
    if (stepTicks > 0) { // skip the first tick (t = 0.01)
      if (memory.totalSize >= learningInfo.batchSize) {
        learner.update(memory.sample(learningInfo.batchSize))
      }
      if (stepTicks % learningInfo.episodeSize == 0) {
        val allExperiences = managers.map { m =>
          if (m.has("experiences")) m.get[Queue[Step]]("experiences")
          else Queue.empty
        }
        learner.endEpisode(allExperiences)
        val newPosition = createPositions()
        environment.getSimulation.schedule { () =>
          agents.foreach(node => environment.removeNode(node))
          newPosition.zip(initialSnapshot).foreach { case (position, prototype) =>
            replaceNodes(position, prototype, currentTime.plus(new DoubleTime(1)))
          }
          episodeCounter += 1
          cachedSourceKey = (-1, Set.empty)
        }
        learner.decayingParameters.foreach { case (_, parameter) => parameter.update() }
      }
    }
    stepTicks += 1
    recordSpatialSnapshotIfFinalEpisode(currentTime.toDouble)
    recordGNNSpatialEmbeddingsIfFinalEpisode(currentTime.toDouble)
  }

  private def recordGNNSpatialEmbeddingsIfFinalEpisode(currentTime: Double): Unit = {
    learner match {
      case gnnLearner: it.unibo.learning.agents.OfflineLearnerGNN if learningInfo.randomSeed == 2 =>
        val w =
          if (managers.nonEmpty && managers.head.has(Sensors.weight)) managers.head.get[Double](Sensors.weight)
          else 0.95
        if (math.abs(w - 0.95) < 1e-6 && currentTime >= 3885.0 && currentTime <= 3960.5) {
          val tEp = currentTime - 3800.0
          val nodePosById = agents.map(n => n.getId -> environment.getPosition(n)).toMap
          agents.foreach { n =>
            val m = new SimpleNodeManager[T](n)
            val pos = nodePosById(n.getId)
            val x = pos.getCoordinate(0)
            val y = pos.getCoordinate(1)
            val isSrc = if (m.has(Sensors.source) && m.get[Boolean](Sensors.source)) 1 else 0
            val g =
              if (m.has(Sensors.localComputation)) m.get[Double](Sensors.localComputation) else Double.PositiveInfinity
            val gFin = if (g.isFinite) g else 600.0
            val gt = groundTruthGradient(n.getId)
            val gtFin = if (gt.isFinite) gt else 600.0
            val dt = if (m.has(Sensors.nextWakeUp)) m.get[Double](Sensors.nextWakeUp) else 5.0
            val err = math.abs(gFin - gtFin)
            val windowComp =
              if (m.has(Sensors.localComputationWindow))
                m.get[Queue[Map[Int, it.unibo.learning.abstractions.AgentState.NeighborInfo]]](
                  Sensors.localComputationWindow
                )
              else Queue.empty
            val windowSens =
              if (m.has(Sensors.windowSensing))
                m.get[Queue[Map[Int, it.unibo.learning.abstractions.AgentState.NeighborInfo]]](Sensors.windowSensing)
              else Queue.empty
            if (windowComp.nonEmpty) {
              val st = new AgentState(n.getId, windowComp, windowSens)
              val inspection = gnnLearner.inspectEmbeddingAndAttention(st, w)
              val (emb, alphas, nbrIds) = (inspection.embedding, inspection.attention, inspection.neighbours)
              // Compute attention-weighted direction vector (attDx, attDy) and max-attended neighbor coordinates
              var attDx = 0.0
              var attDy = 0.0
              var maxAlpha = 0.0f
              var bestNbrX = x
              var bestNbrY = y
              var entropy = 0.0
              nbrIds.zipWithIndex.foreach { case (nid, k) =>
                val a = alphas(k)
                if (a > 1e-6f) entropy -= a * math.log(a)
                nodePosById.get(nid).foreach { nPos =>
                  val dx = nPos.getCoordinate(0) - x
                  val dy = nPos.getCoordinate(1) - y
                  attDx += a * dx
                  attDy += a * dy
                  if (a > maxAlpha) {
                    maxAlpha = a
                    bestNbrX = nPos.getCoordinate(0)
                    bestNbrY = nPos.getCoordinate(1)
                  }
                }
              }
              val embStr = emb.take(16).map(v => f"$v%.4f").mkString(",")
              gnnSpatialRows += f"$tEp%.1f,${n.getId},$x%.2f,$y%.2f,$isSrc,$gFin%.2f,$gtFin%.2f,$dt%.4f,$err%.2f,$maxAlpha%.4f,$entropy%.4f,$attDx%.2f,$attDy%.2f,$bestNbrX%.2f,$bestNbrY%.2f,$embStr"
            }
          }
          if (currentTime >= 3959.5 && gnnSpatialRows.nonEmpty) {
            val outDir = new java.io.File("build/exports/spatial")
            outDir.mkdirs()
            val pw = new java.io.PrintWriter(new java.io.File(outDir, "gnn_spatiotemporal_embeddings.csv"))
            try {
              val hCols = (0 until 16).map(i => s"h_$i").mkString(",")
              pw.println(
                s"tEp,nodeId,x,y,isSource,gradient,groundTruth,nextWakeUp,error,maxAlpha,attEntropy,attDx,attDy,bestNbrX,bestNbrY,$hCols"
              )
              gnnSpatialRows.foreach(pw.println)
            } finally pw.close()
          }
        }
      case _ =>
    }
  }

  private def recordSpatialSnapshotIfFinalEpisode(currentTime: Double): Unit = {
    if (learningInfo.randomSeed != 2 || learner.getClass.getSimpleName != "OfflineLearnerContinuous") return
    if (currentTime >= 3802.0 && currentTime <= 4000.5) {
      val w =
        if (managers.nonEmpty && managers.head.has(Sensors.weight)) managers.head.get[Double](Sensors.weight) else 0.0
      if (math.abs(w - 0.8) < 1e-6 || math.abs(w - 0.95) < 1e-6) {
        if (spatialEdges.isEmpty) {
          agents.foreach { u =>
            val posU = environment.getPosition(u)
            environment.getNeighborhood(u).getNeighbors.iterator().asScala.foreach { v =>
              if (u.getId < v.getId) {
                val posV = environment.getPosition(v)
                spatialEdges += f"${u.getId},${v.getId},${posU.getCoordinate(0)}%.2f,${posU.getCoordinate(1)}%.2f,${posV
                    .getCoordinate(0)}%.2f,${posV.getCoordinate(1)}%.2f"
              }
            }
          }
        }
        val tEp = currentTime - 3800.0
        agents.foreach { n =>
          val m = new SimpleNodeManager[T](n)
          val pos = environment.getPosition(n)
          val x = pos.getCoordinate(0)
          val y = pos.getCoordinate(1)
          val isSrc = if (m.has(Sensors.source) && m.get[Boolean](Sensors.source)) 1 else 0
          val g =
            if (m.has(Sensors.localComputation)) m.get[Double](Sensors.localComputation) else Double.PositiveInfinity
          val gFin = if (g.isFinite) g else 600.0
          val gt = groundTruthGradient(n.getId)
          val gtFin = if (gt.isFinite) gt else 600.0
          val dt = if (m.has(Sensors.nextWakeUp)) m.get[Double](Sensors.nextWakeUp) else 5.0
          val err = math.abs(gFin - gtFin)
          val tks = if (m.has(Sensors.ticks)) m.get[Number](Sensors.ticks).intValue() else 0
          spatialRows += f"$tEp%.2f,${n.getId},$x%.2f,$y%.2f,$isSrc,$gFin%.2f,$gtFin%.2f,$dt%.4f,$err%.2f,$tks"
        }
        if (currentTime >= 3999.0) {
          val outDir = new java.io.File("build/exports/spatial")
          outDir.mkdirs()
          val pw = new java.io.PrintWriter(new java.io.File(outDir, s"spatial_snapshots_weight-$w.csv"))
          try {
            pw.println("tEp,nodeId,x,y,isSource,gradient,groundTruth,nextWakeUp,error,ticks")
            spatialRows.foreach(pw.println)
          } finally pw.close()
          val ew = new java.io.PrintWriter(new java.io.File(outDir, s"spatial_edges_weight-$w.csv"))
          try {
            ew.println("u,v,xU,yU,xV,yV")
            spatialEdges.foreach(ew.println)
          } finally ew.close()
        }
      }
    }
  }

  override def initializationComplete(time: Time, environment: Environment[T, _]): Unit =
    initialSnapshot = agents.map(_.cloneNode(Time.ZERO))

  def decayVariableOf(name: String): Double =
    learner.decayingParameters.collectFirst { case (`name`, parameter) => parameter.value }.getOrElse(0.0)

  /** During training (first 10 episodes) always uses "gradient"; during exploitation (episodes 10+) switches to
    * targetProgram for zero-shot transfer.
    */
  def activeProgramName(targetProgram: String): String =
    if (episodeCounter < 10) "gradient" else targetProgram

  def programScale(programName: String): Double = programName match {
    case "hopGradient" => 10.0
    case "broadcast" => 500.0
    case "collect" => 40.0
    case _ => 500.0
  }

  /** Computes the exact ground truth for any supported self-stabilizing building block ("gradient", "hopGradient",
    * "broadcast", "collect").
    */
  def groundTruthGradient(nodeId: Int, programName: String = "gradient"): Double = {
    val currentSources = agents
      .filter { n =>
        val m = new SimpleNodeManager[T](n)
        m.has(Sensors.source) && m.get[Boolean](Sensors.source)
      }
      .map(_.getId)
      .toSet

    val key = (episodeCounter, currentSources)
    if (key != cachedSourceKey) {
      cachedSourceKey = key
      cachedGroundTruth = computeShortestPaths(currentSources)
      cachedHopTruth = computeHopShortestPaths(currentSources)
      cachedCollectTruth = computeCollectSubtreeCounts(cachedGroundTruth)
      cachedBroadcastTruth = currentSources.headOption.map(id => (id + 1).toDouble * 10.0).getOrElse(0.0)
    }
    programName match {
      case "hopGradient" => cachedHopTruth.getOrElse(nodeId, Double.PositiveInfinity)
      case "broadcast" => if (currentSources.nonEmpty) cachedBroadcastTruth else 0.0
      case "collect" => cachedCollectTruth.getOrElse(nodeId, 1.0)
      case _ => cachedGroundTruth.getOrElse(nodeId, Double.PositiveInfinity)
    }
  }

  private def computeHopShortestPaths(sources: Set[Int]): Map[Int, Double] = {
    if (sources.isEmpty) return Map.empty
    val dist = scala.collection.mutable.Map.empty[Int, Double].withDefaultValue(Double.PositiveInfinity)
    val q = scala.collection.mutable.Queue.empty[Node[T]]
    val nodeById = agents.map(n => n.getId -> n).toMap
    sources.flatMap(nodeById.get).foreach { srcNode =>
      dist(srcNode.getId) = 0.0
      q.enqueue(srcNode)
    }
    while (q.nonEmpty) {
      val u = q.dequeue()
      val d = dist(u.getId)
      environment.getNeighborhood(u).getNeighbors.iterator().asScala.foreach { v =>
        if (d + 1.0 < dist(v.getId)) {
          dist(v.getId) = d + 1.0
          q.enqueue(v)
        }
      }
    }
    dist.toMap
  }

  private def computeCollectSubtreeCounts(potentials: Map[Int, Double]): Map[Int, Double] = {
    if (potentials.isEmpty) return Map.empty
    val counts = scala.collection.mutable.Map.empty[Int, Double]
    agents.foreach(n => counts(n.getId) = 1.0)
    val sortedNodes = agents.sortBy(n => -potentials.getOrElse(n.getId, Double.PositiveInfinity))
    sortedNodes.foreach { u =>
      val potU = potentials.getOrElse(u.getId, Double.PositiveInfinity)
      if (potU > 0.0 && potU.isFinite) {
        val neighbors = environment.getNeighborhood(u).getNeighbors.iterator().asScala.toList
        val parentOpt = neighbors
          .filter(v => potentials.getOrElse(v.getId, Double.PositiveInfinity) < potU)
          .minByOption(v => (potentials.getOrElse(v.getId, Double.PositiveInfinity), v.getId))
        parentOpt.foreach { p =>
          counts(p.getId) = counts(p.getId) + counts(u.getId)
        }
      }
    }
    counts.toMap
  }

  private def computeShortestPaths(sources: Set[Int]): Map[Int, Double] = {
    if (sources.isEmpty) return Map.empty
    val dist = scala.collection.mutable.Map.empty[Int, Double].withDefaultValue(Double.PositiveInfinity)
    val pq = scala.collection.mutable.PriorityQueue.empty[(Double, Node[T])](
      Ordering.by[(Double, Node[T]), Double](-_._1)
    )
    val nodeById = agents.map(n => n.getId -> n).toMap
    sources.flatMap(nodeById.get).foreach { srcNode =>
      dist(srcNode.getId) = 0.0
      pq.enqueue((0.0, srcNode))
    }
    while (pq.nonEmpty) {
      val (d, u) = pq.dequeue()
      if (d <= dist(u.getId)) {
        val posU = environment.getPosition(u)
        environment.getNeighborhood(u).getNeighbors.iterator().asScala.foreach { v =>
          val weight = posU.distanceTo(environment.getPosition(v))
          val alt = d + weight
          if (alt < dist(v.getId)) {
            dist(v.getId) = alt
            pq.enqueue((alt, v))
          }
        }
      }
    }
    dist.toMap
  }
  private def createPositions() = {
    val gridRng =
      new org.apache.commons.math3.random.Well19937c(learningInfo.randomSeed.toLong * 1000L + episodeCounter + 1L)
    new Grid(
      environment,
      gridRng,
      0,
      0,
      environmentBox.width,
      environmentBox.width,
      environmentBox.step,
      environmentBox.step,
      environmentBox.randomness,
      environmentBox.randomness
    ).iterator().asScala.toList
  }

  private def replaceNodes(
      position: Position[_],
      prototype: Node[T],
      currentTime: Time
  ): Unit = {
    val newNode = prototype.cloneNode(currentTime)
    newNode.getReactions.iterator().asScala.toList.foreach(r => newNode.removeReaction(r))
    newNode.getContents.asScala.toList.foreach { case (molecule, _) => newNode.removeConcentration(molecule) }
    environment.addNode(newNode, position.asInstanceOf[P])
    prototype.getContents.asScala.foreach { case (molecule, content) =>
      newNode.setConcentration(molecule, content)
    }
    prototype.getReactions
      .iterator()
      .asScala
      .foreach(reaction => newNode.addReaction(reaction.cloneOnNewNode(newNode, currentTime)))
  }

}
