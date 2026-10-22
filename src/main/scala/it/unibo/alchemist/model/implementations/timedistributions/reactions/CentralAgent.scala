package it.unibo.alchemist.model.implementations.timedistributions.reactions

import it.unibo.alchemist.Molecules._
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent._
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces._
import it.unibo.learning.abstractions.{AgentState, ReplayBuffer, Step}
import it.unibo.learning.agents.Learner
import it.unibo.learning.{Box, LearningInfo}
import it.unibo.scafi.{BuildingBlock, EvaluationProtocol, GroundTruths, Sensors, Topology}

import scala.collection.immutable.Queue
import scala.jdk.CollectionConverters._
import scala.util.Random

/** The global reaction that drives the experiment: it trains the shared learner (online at every step, offline at the
  * end of each episode), starts each episode on a new deployment, provides the ground truths used to measure the nodes'
  * error, and records the final episode of the inspected run.
  */
final class CentralAgent[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    learningInfo: LearningInfo,
    learner: Learner,
    environmentBox: Box
) extends AbstractGlobalReaction[T, P](environment, distribution) {
  private val randomness = new Random(learningInfo.randomSeed)
  private val redeployment = new Redeployment(environment, environmentBox, learningInfo.randomSeed)
  private val snapshots = new FinalEpisodeSnapshots(environment, learner, learningInfo, groundTruth(_))
  private var steps: Int = 0
  private var episode: Int = 0

  /** The ground truths of the current episode, recomputed when the sources change. */
  private var groundTruthsCache: Option[(Int, GroundTruths)] = None

  /** The experience of all the nodes, for the learners that train online. */
  val memory: ReplayBuffer = new ReplayBuffer(learningInfo.bufferSize, randomness)
  learner.injectRandom(randomness)

  override def initializationComplete(time: Time, environment: Environment[T, _]): Unit =
    redeployment.capturePrototypes()

  override def executeBeforeUpdateDistribution(): Unit = {
    val now = environment.getSimulation.getTime
    if (steps > 0) { // the first step happens at the very beginning, before any round
      trainOnline()
      if (steps % learningInfo.episodeSize == 0) endEpisode(now)
    }
    steps += 1
    snapshots.observe(now.toDouble)
  }

  private def trainOnline(): Unit =
    if (memory.totalSize >= learningInfo.batchSize) learner.update(memory.sample(learningInfo.batchSize))

  private def endEpisode(now: Time): Unit = {
    learner.endEpisode(agents.map(_.valueOf[Queue[Step]](Sensors.experiences).getOrElse(Queue.empty)))
    val positions = redeployment.positionsFor(episode)
    environment.getSimulation.schedule { () =>
      redeployment.redeploy(positions, now.plus(new DoubleTime(RestartDelay)))
      episode += 1
    }
    learner.decayingParameters.foreach { case (_, parameter) => parameter.update() }
  }

  /** The policy of the learner; in the generalization scenario, conditioned on the weight under evaluation. */
  def policy: AgentState => Double =
    EvaluationProtocol
      .evaluationWeightAt(episode)
      .filter(_ => isGeneralizationScenario)
      .fold(learner.policy)(learner.policyForWeight)

  private def isGeneralizationScenario: Boolean = agents.headOption.exists(_.valueOf[Any](Sensors.program).nonEmpty)

  /** The current value of the decaying hyper-parameter `name` of the learner (0 if it has none). */
  def decayVariableOf(name: String): Double =
    learner.decayingParameters.collectFirst { case (`name`, parameter) => parameter.value }.getOrElse(0.0)

  /** The building block the nodes run in the current episode, given the one they are configured for. */
  def activeBlock(target: BuildingBlock): BuildingBlock = EvaluationProtocol.blockAt(episode, target)

  /** The value `block` should converge to in node `nodeId`, given the current sources. */
  def groundTruth(nodeId: Int, block: BuildingBlock = BuildingBlock.Gradient): Double =
    currentGroundTruths.of(nodeId, block)

  private def currentGroundTruths: GroundTruths = {
    val sources = currentSources
    groundTruthsCache
      .collect { case (cachedEpisode, truths) if cachedEpisode == episode && truths.sources == sources => truths }
      .getOrElse {
        val truths = GroundTruths.of(topology, sources)
        groundTruthsCache = Some((episode, truths))
        truths
      }
  }

  private def currentSources: Set[Int] =
    agents.filter(_.valueOf[Boolean](Sensors.source).getOrElse(false)).map(_.getId).toSet

  private def topology: Topology = {
    def linksOf(node: Node[T]): Seq[(Int, Double)] = {
      val position = environment.getPosition(node)
      environment.getNeighborhood(node).getNeighbors.iterator().asScala.toSeq.map { neighbour =>
        neighbour.getId -> position.distanceTo(environment.getPosition(neighbour))
      }
    }
    Topology(agents.map(_.getId), agents.map(node => node.getId -> linksOf(node)).toMap)
  }
}

object CentralAgent {

  /** The nodes of a new episode start this long after the end of the previous one. */
  val RestartDelay: Double = 1.0
}
