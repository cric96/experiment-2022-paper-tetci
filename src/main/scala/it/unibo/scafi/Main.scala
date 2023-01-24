package it.unibo.scafi

import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.learning.abstractions.AgentState.NeighborInfo
import it.unibo.learning.abstractions.{AgentState, Contextual, ReplayBuffer}
import it.unibo.learning.network.RDQN
import it.unibo.learning.network.torch.{Pack, torch}

import scala.collection.immutable.Queue
import scala.jdk.CollectionConverters.IteratorHasAsScala

class Main
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with BlockG
    with Gradients
    with FieldUtils
    with StateManagement {

  lazy val localWindowSize = node.getOption("window").getOrElse(3)
  lazy val actionSpace = node.getOption("actions").getOrElse(List(1.0, 1.5, 2, 3))
  lazy val sharedMemory: ReplayBuffer = loadMemory()
  lazy val weightForConvergence = 0.01
  def policy: (AgentState => (Int, Contextual)) = loadPolicy()

  override def main(): Any = {
    val localComputation = computation()
    node.put("localComputation", localComputation)
    val localSensing = perception()
    val (_, _, Some(action)) = rep((Option.empty[AgentState], (), Option.empty[Int])) {
      case (oldState, oldContext, oldAction) =>
        val fieldComputation =
          includingSelf.reifyField(NeighborInfo(nbr(localComputation), nbrRange(), nbr(oldAction).getOrElse(-1)))
        val fieldSensing =
          includingSelf.reifyField(NeighborInfo(nbr(localSensing), nbrRange(), nbr(oldAction).getOrElse(-1)))

        val windowComputation = window(fieldComputation)
        val windowFieldSensing = window(fieldSensing)

        val state = new AgentState(mid(), windowComputation, windowFieldSensing, oldContext)
        val reward = evalReward(state, oldAction)
        val (action, context) = policy(state)
        oldState.zip(oldAction).foreach { case (stateT, actionT) =>
          node.put("reward", reward)
          sharedMemory.put(stateT, actionT, reward, state)
        }
        node.put("next-wake-up", actionSpace(action)) // Actuation
        node.put("field-computation", fieldComputation)
        node.put("local-computation-window", windowComputation.map(_.get(mid())))
        node.put("field-sensing", fieldSensing)
        node.put("window-computation", windowComputation)
        node.put("window-sensing", windowFieldSensing)
        node.put("ticks", roundCounter())
        (Some(state), context, Some(action))
    }
  }

  def computation(): Double = classicGradient(sense("source"))
  def perception(): Double = 0.0 // what I need from my neighborhood for computing the state

  def window(snapshot: Map[ID, NeighborInfo]): Queue[Map[ID, NeighborInfo]] =
    rep(Queue.empty[Map[ID, NeighborInfo]])(queue => (queue :+ snapshot).takeRight(localWindowSize))

  def loadPolicy(): (AgentState => (Int, Contextual)) = global().policy

  def loadMemory(): ReplayBuffer = global().memory

  def evalReward(state: AgentState, oldAction: Option[Int]): Double = {
    val myOutput = state.neighborhoodOutput.map(neigh => neigh(state.me))
    val history = myOutput
      .map(_.data)
      .sliding(2, 1)
      .toList
      .map(x =>
        if (x.size == 1) { 0.0 }
        else { (x.last - x.head).sign }
      )
    node.put("history", history)
    if (history.headOption.exists(_ != 0.0)) {
      -weightForConvergence * ((deltaTime.toMillis / 1000.0) / actionSpace.max)
    } else {
      -(1 - ((deltaTime.toMillis / 1000.0) / actionSpace.max)) * (1 - weightForConvergence)
    }
  }

  def global(): CentralAgent[_, _] = alchemistEnvironment.getGlobalReactions
    .iterator()
    .asScala
    .collectFirst { case reaction: CentralAgent[_, _] => reaction }
    .get
}
