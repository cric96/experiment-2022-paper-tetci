package it.unibo.scafi

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.learning.abstractions.AgentState.NeighborInfo
import it.unibo.learning.abstractions.{AgentState, Contextual, ReplayBuffer}
import it.unibo.util.TemporalInfo

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

  import Sensors._

  private lazy val localWindowSize = node.getOption(Sensors.window).getOrElse(3)
  private lazy val actionSpace = node.getOption(actions).getOrElse(List(1.0, 1.5, 2, 3))
  private lazy val sharedMemory: ReplayBuffer = loadMemory()
  private lazy val weightForConvergence = node.getOption(Sensors.weight).getOrElse(0.9)
  def policy: (AgentState => (Int, Contextual)) = loadPolicy()

  override def main(): Any = {
    val localComputation = computation()
    val fullSpeed = node.get[Boolean](Sensors.fullSpeed)
    branch(!fullSpeed) {
      update(localComputation)
    } {
      node.put(Sensors.groundTruth, localComputation)
      exec()
    }
  }

  def update(localComputation: Double): Unit = {
    node.put(localComputationWindow, localComputation)
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
          node.put(Sensors.reward, reward)
          sharedMemory.put(stateT, actionT, reward, state)
        }
        node.put(Sensors.nextWakeUp, actionSpace(action)) // Actuation
        node.put(Sensors.fieldComputation, fieldComputation)
        node.put(Sensors.localComputationWindow, windowComputation.map(_.get(mid())))
        node.put(Sensors.fieldSensing, fieldSensing)
        node.put(Sensors.localComputationWindow, windowComputation)
        node.put(Sensors.windowSensing, windowFieldSensing)
        node.put(Sensors.ticks, roundCounter())
        val me = alchemistEnvironment.getPosition(alchemistEnvironment.getNodeByID(mid()))
        val fastestResult =
          alchemistEnvironment
            .getNodesWithinRange(me, 0.01)
            .iterator()
            .asScala
            .toList
            .filter(_.getId != mid())
            .map(new SimpleNodeManager[Any](_))
            .map(_.get[Double](Sensors.groundTruth))
            .head
        node.put(Sensors.error, math.abs(localComputation.convertIfInfinite - fastestResult.convertIfInfinite))
        (Some(state), context, Some(action))
    }
  }

  def exec(): Unit =
    node.put(Sensors.nextWakeUp, actionSpace.min)

  def computation(): Double = classicGradient(sense(Sensors.source))
  def perception(): Double = 0.0 // what I need from my neighborhood for computing the state

  def window(snapshot: Map[ID, NeighborInfo]): Queue[Map[ID, NeighborInfo]] =
    rep(Queue.empty[Map[ID, NeighborInfo]])(queue => (queue :+ snapshot).takeRight(localWindowSize))

  def loadPolicy(): (AgentState => (Int, Contextual)) = global().policy

  def loadMemory(): ReplayBuffer = global().memory

  def evalReward(state: AgentState, oldAction: Option[Int]): Double = {
    val myOutput = state.neighborhoodOutput.map(neigh => neigh(state.me))
    val history = TemporalInfo.computeDeltaTrend(myOutput.map(_.data))

    node.put(Sensors.history, history)
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
