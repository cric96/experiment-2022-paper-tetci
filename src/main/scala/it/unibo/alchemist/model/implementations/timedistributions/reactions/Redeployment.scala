package it.unibo.alchemist.model.implementations.timedistributions.reactions

import it.unibo.alchemist.loader.deployments.Grid
import it.unibo.alchemist.model.implementations.timedistributions.reactions.Redeployment._
import it.unibo.alchemist.model.interfaces.{Environment, Node, Position, Time}
import it.unibo.learning.Box
import org.apache.commons.math3.random.Well19937c

import scala.jdk.CollectionConverters._

/** Starts every episode on a fresh, randomly perturbed grid of nodes that are clones of the initial ones. */
final class Redeployment[T, P <: Position[P]](environment: Environment[T, P], box: Box, runSeed: Int) {
  private var prototypes: List[Node[T]] = List.empty

  /** Remembers the nodes as they are at the beginning of the simulation. */
  def capturePrototypes(): Unit = prototypes = nodes.map(_.cloneNode(Time.ZERO))

  /** The deployment of `episode`, reproducible for a given run. */
  def positionsFor(episode: Int): List[Position[_]] = {
    val random = new Well19937c(runSeed.toLong * SeedMultiplier + episode + 1L)
    new Grid(environment, random, 0, 0, box.width, box.width, box.step, box.step, box.randomness, box.randomness)
      .iterator()
      .asScala
      .toList
  }

  /** Replaces the current nodes with fresh clones of the prototypes in `positions`, whose reactions start at `start`. */
  def redeploy(positions: List[Position[_]], start: Time): Unit = {
    nodes.foreach(environment.removeNode)
    positions.zip(prototypes).foreach { case (position, prototype) => deployClone(prototype, position, start) }
  }

  private def deployClone(prototype: Node[T], position: Position[_], start: Time): Unit = {
    val node = prototype.cloneNode(start)
    node.getReactions.asScala.toList.foreach(node.removeReaction)
    node.getContents.asScala.keys.toList.foreach(node.removeConcentration)
    // The grid deploys in the position type of the environment.
    environment.addNode(node, position.asInstanceOf[P])
    prototype.getContents.asScala.foreach { case (molecule, concentration) =>
      node.setConcentration(molecule, concentration)
    }
    prototype.getReactions.asScala.foreach(reaction => node.addReaction(reaction.cloneOnNewNode(node, start)))
  }

  private def nodes: List[Node[T]] = environment.getNodes.iterator().asScala.toList
}

object Redeployment {

  /** Spreads the run seed, so that the deployments of different runs do not overlap. */
  val SeedMultiplier: Long = 1000L
}
