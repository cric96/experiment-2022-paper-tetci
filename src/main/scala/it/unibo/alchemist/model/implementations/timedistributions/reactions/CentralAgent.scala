package it.unibo.alchemist.model.implementations.timedistributions.reactions

import it.unibo.alchemist.loader.deployments.Grid
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces._
import it.unibo.learning.{Box, LearningInfo}
import it.unibo.learning.abstractions.{DecayReference, ReplayBuffer}
import it.unibo.learning.agents.Learner
import it.unibo.learning.network.torch
import org.apache.commons.math3.random.RandomGenerator

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
  private var references: List[(String, DecayReference[Double])] = List.empty
  private var averagedNextWakeUp = 0.0
  private var averageRewardPerEpisode = 0.0
  // used to restart the simulation with the same configuration
  private var initialSnapshot: List[Node[T]] =
    List.empty[Node[T]]
  val memory: ReplayBuffer = new ReplayBuffer(learningInfo.bufferSize, scalaRandom)
  learner.injectCentralAgent(this)
  learner.injectRandom(scalaRandom)

  def attachDecayable(elements: (String, DecayReference[Double])*) = references = references ::: (elements.toList)

  def policy = learner.policy

  override def executeBeforeUpdateDistribution(): Unit = {
    val currentTime = environment.getSimulation.getTime
    if (currentTime.toDouble < 1) {
      filterFullSpeed.foreach { case (node, position) => replaceNodes(position, node, currentTime) }
    }
    val sample = memory.sample(learningInfo.batchSize)
    if (currentTime.toDouble > 1 && sample.size == learningInfo.batchSize) { // skip the first tick
      learner.update(memory.sample(learningInfo.batchSize))
      if (currentTime.toDouble.toInt % learningInfo.episodeSize == 0) {
        val newPosition = createPositions()
        agents.foreach(node => environment.removeNode(node))
        clonePutSpeed(newPosition, currentTime)
        newPosition.zip(initialSnapshot).foreach { case (position, prototype) =>
          replaceNodes(position, prototype, currentTime)
        }
        references.foreach { case (name, value) =>
          torch.writer.add_scalar(name, value.value, environment.getSimulation.getStep)
        }
        references.foreach(_._2.update())
        // logging phase
        torch.writer.add_scalar(
          "total-average",
          averagedNextWakeUp / learningInfo.episodeSize,
          environment.getSimulation.getStep
        )
        torch.writer.add_scalar(
          "reward-average",
          averageRewardPerEpisode / learningInfo.episodeSize,
          environment.getSimulation.getStep
        )
        averagedNextWakeUp = 0
        averageRewardPerEpisode = 0
      }
    }
    val consumption =
      managers.filterNot(_.get[Boolean]("full")).map(_.get[Double]("next-wake-up")).sum / (managers.size / 2)
    val rewardAverage =
      managers.filterNot(_.get[Boolean]("full")).map(_.get[Double]("reward")).sum / (managers.size / 2)
    val errorAverage = managers.filterNot(_.get[Boolean]("full")).map(_.get[Double]("error")).sum / (managers.size / 2)
    averageRewardPerEpisode += rewardAverage
    averagedNextWakeUp += consumption
    torch.writer.add_scalar("average-wake-up-time", consumption, environment.getSimulation.getStep)
    torch.writer.add_scalar("reward", rewardAverage, environment.getSimulation.getStep)
    torch.writer.add_scalar("error", errorAverage, environment.getSimulation.getStep)
  }

  override def initializationComplete(time: Time, environment: Environment[T, _]): Unit =
    initialSnapshot = agents.map(_.cloneNode(Time.ZERO))

  private def createPositions() = new Grid(
    environment,
    random,
    0,
    0,
    environmentBox.width,
    environmentBox.width,
    environmentBox.step,
    environmentBox.step,
    environmentBox.randomness,
    environmentBox.randomness
  ).iterator().asScala.toList

  private def replaceNodes(
      position: Position[_],
      prototype: Node[T],
      currentTime: Time,
      full: Boolean = false
  ): Unit = {
    val newNode = prototype.cloneNode(currentTime)
    newNode.getReactions.iterator().asScala.toList.foreach(r => newNode.removeReaction(r))
    newNode.getContents.asScala.toList.foreach { case (molecule, _) => newNode.removeConcentration(molecule) }
    environment.addNode(newNode, position.asInstanceOf[P])
    prototype.getContents.asScala.foreach { case (molecule, content) =>
      newNode.setConcentration(molecule, content)
    }
    newNode.setConcentration(new SimpleMolecule("full"), full.asInstanceOf[T])
    prototype.getReactions
      .iterator()
      .asScala
      .foreach(reaction => newNode.addReaction(reaction.cloneOnNewNode(newNode, currentTime)))
  }

  private def clonePutSpeed(position: List[Position[_]], time: Time): Unit = {
    val cloned = initialSnapshot.zip(position)
    cloned.foreach { case (node, position) => replaceNodes(position, node, time, full = true) }
  }

  private def filterFullSpeed = environment.getNodes
    .iterator()
    .asScala
    .toList
    .map(node => new SimpleNodeManager[T](node))
    .filter(manager => manager.get[Boolean]("full"))
    .map(manager => (manager.node, environment.getPosition(manager.node)))
}

object CentralAgent {
  case class LearningInfo(randomSeed: Int, bufferSize: Int, batchSize: Int, episodeSize: Int)
}
