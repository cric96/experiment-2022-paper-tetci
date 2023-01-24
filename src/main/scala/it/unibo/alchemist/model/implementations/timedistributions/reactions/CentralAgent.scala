package it.unibo.alchemist.model.implementations.timedistributions.reactions

import it.unibo.alchemist.loader.deployments.Grid
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
  val scalaRandom = new Random(learningInfo.randomSeed)
  private var references: List[(String, DecayReference[Double])] = List.empty
  private var averagedNextWakeUp = 0.0
  private var initialSnapshot: List[Node[T]] =
    List.empty[Node[T]] // used to restart the simulation with the same configuration
  val memory: ReplayBuffer = new ReplayBuffer(learningInfo.bufferSize, scalaRandom)
  learner.injectCentralAgent(this)
  learner.injectRandom(scalaRandom)

  def attachDecayable(elements: (String, DecayReference[Double])*) = references = references ::: (elements.toList)

  def policy = learner.policy

  override def executeBeforeUpdateDistribution(): Unit = {
    val currentTime = environment.getSimulation.getTime
    val sample = memory.sample(learningInfo.batchSize)
    if (currentTime.toDouble > 1 && sample.size == learningInfo.batchSize) { // skip the first tick
      learner.update(memory.sample(learningInfo.batchSize))
      if (currentTime.toDouble.toInt % learningInfo.episodeSize == 0) {
        torch.writer.add_scalar(
          "total-average",
          averagedNextWakeUp / learningInfo.episodeSize,
          environment.getSimulation.getStep
        )
        averagedNextWakeUp = 0
        val newPosition = new Grid(
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
        agents.foreach(node => environment.removeNode(node))
        newPosition.zip(initialSnapshot).foreach { case (position, prototype) =>
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
        references.foreach { case (name, value) =>
          torch.writer.add_scalar(name, value.value, environment.getSimulation.getStep)
        }
        references.foreach(_._2.update())
      }
    }
    val consumption = managers.map(_.get[Double]("next-wake-up")).sum / managers.size
    val rewardAverage = managers.map(_.get[Double]("reward")).sum / managers.size
    averagedNextWakeUp += consumption
    torch.writer.add_scalar("average-wake-up-time", consumption, environment.getSimulation.getStep)
    torch.writer.add_scalar("reward", rewardAverage, environment.getSimulation.getStep)
  }

  override def initializationComplete(time: Time, environment: Environment[T, _]): Unit =
    initialSnapshot = agents.map(_.cloneNode(Time.ZERO))
}

object CentralAgent {
  case class LearningInfo(randomSeed: Int, bufferSize: Int, batchSize: Int, episodeSize: Int)
}
