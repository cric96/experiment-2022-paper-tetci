package it.unibo.alchemist.model.implementations.timedistributions.reactions

import it.unibo.alchemist.loader.deployments.Grid
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces._
import it.unibo.learning.{Box, LearningInfo}
import it.unibo.learning.abstractions.{AgentState, ReplayBuffer, Step}
import it.unibo.learning.agents.Learner
import it.unibo.scafi.{BuildingBlock, EvaluationProtocol, GroundTruths, Sensors, Topology}
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

  /** The ground truths of the current episode, recomputed when the sources change. */
  private var groundTruthsCache: Option[(Int, GroundTruths)] = None
  private val spatialRows = scala.collection.mutable.ArrayBuffer.empty[String]
  private val spatialEdges = scala.collection.mutable.ArrayBuffer.empty[String]
  private val gnnSpatialRows = scala.collection.mutable.ArrayBuffer.empty[String]
  // used to restart the simulation with the same configuration
  private var initialSnapshot: List[Node[T]] =
    List.empty[Node[T]]
  val memory: ReplayBuffer = new ReplayBuffer(learningInfo.bufferSize, scalaRandom)
  learner.injectRandom(scalaRandom)

  /** The policy of the learner; in the generalization scenario, conditioned on the weight under evaluation. */
  def policy: AgentState => Double =
    EvaluationProtocol
      .evaluationWeightAt(episodeCounter)
      .filter(_ => isGeneralizationScenario)
      .fold(learner.policy)(learner.policyForWeight)

  private def isGeneralizationScenario: Boolean = managers.headOption.exists(_.has(Sensors.program))

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
            val gt = groundTruth(n.getId)
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
          val gt = groundTruth(n.getId)
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

  /** The building block the nodes run in the current episode, given the one they are configured for. */
  def activeBlock(target: BuildingBlock): BuildingBlock = EvaluationProtocol.blockAt(episodeCounter, target)

  /** The value `block` should converge to in node `nodeId`, given the current sources. */
  def groundTruth(nodeId: Int, block: BuildingBlock = BuildingBlock.Gradient): Double =
    currentGroundTruths.of(nodeId, block)

  private def currentGroundTruths: GroundTruths = {
    val sources = currentSources
    groundTruthsCache
      .collect { case (episode, truths) if episode == episodeCounter && truths.sources == sources => truths }
      .getOrElse {
        val truths = GroundTruths.of(topology, sources)
        groundTruthsCache = Some((episodeCounter, truths))
        truths
      }
  }

  private def currentSources: Set[Int] =
    managers
      .filter(manager => manager.has(Sensors.source) && manager.get[Boolean](Sensors.source))
      .map(_.node.getId)
      .toSet

  private def topology: Topology = {
    def linksOf(node: Node[T]): Seq[(Int, Double)] = {
      val position = environment.getPosition(node)
      environment.getNeighborhood(node).getNeighbors.iterator().asScala.toSeq.map { neighbour =>
        neighbour.getId -> position.distanceTo(environment.getPosition(neighbour))
      }
    }
    Topology(agents.map(_.getId), agents.map(node => node.getId -> linksOf(node)).toMap)
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
