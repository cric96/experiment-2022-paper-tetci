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
  private val snapshots = new FinalEpisodeSnapshots(environment, learner, learningInfo, groundTruth(_))
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
    snapshots.observe(currentTime.toDouble)
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
