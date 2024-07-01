package it.unibo.scafi

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.learning.abstractions.AgentState.NeighborInfo
import it.unibo.learning.abstractions.{AgentState, Contextual, ReplayBuffer}
import it.unibo.util.TemporalInfo
import it.unibo.alchemist.EnvironmentOps
import scala.collection.immutable.Queue
import scala.concurrent.duration.FiniteDuration
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
  private lazy val resultSize = 3
  private lazy val actionSpace = node.getOption(actions).getOrElse(List(1.0, 1.5, 2, 3))
  private lazy val sharedMemory: ReplayBuffer = loadMemory()
  private lazy val weightForConvergence = node.getOption(Sensors.weight).getOrElse(0.9)
  def policy: (AgentState => (Int, Contextual)) = loadPolicy()

  override def main(): Any = {
    /*val fullSpeed = node.get[Boolean](Sensors.fullSpeed)
    val localComputation = branch(!fullSpeed)(computation())(computation())
    val computationWindow = rep(Queue.empty[Double])(queue => (queue :+ localComputation).takeRight(resultSize))
    if (!fullSpeed) {
      node.put(Sensors.localComputation, localComputation)
    } else {
      node.put(Sensors.groundTruth, localComputation)
    }
    branch(!fullSpeed) {
      update(localComputation, computationWindow)
    } {
      node.put(Sensors.groundTruthWindow, computationWindow)
      exec()
    }*/
    val localComputation = computation()
    val computationWindow = rep(Queue.empty[Double])(queue => (queue :+ localComputation).takeRight(resultSize))
    node.put(Sensors.localComputation, localComputation)
    update(localComputation, computationWindow)
  }

  def update(localComputation: Double, elements: Queue[Double]): Unit = {
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
          sharedMemory.put(stateT, actionT, reward, state)
        }
        val accumulatedReward = rep(0.0)(acc => acc + reward)
        val unstableTime = oldState match {
          case Some(stateT) =>
            stateT.neighborhoodOutput
              .map(neigh => neigh(stateT.me))
              .headOption
              .filter(_.data != localComputation)
              .map(_ => deltaFixed)
              .getOrElse(0.0)
          case None =>
            0.0
        }
        val accumulatedUnstableTime = rep(0.0)(acc => acc + unstableTime)
        node.put("accumulatedUnstableTime", accumulatedUnstableTime)
        // val fastestNode = EnvironmentOps(alchemistEnvironment).getClonedOfThis(mid())
        // val fastestResult =
        //  fastestNode.get[Queue[Double]](Sensors.groundTruthWindow)

        // val filterInfinity = fastestResult.filterNot(_.isInfinity)
        // val localFilterInfinity = elements.filterNot(_.isInfinity)
        // val error = if (filterInfinity.size == resultSize && localFilterInfinity.size == resultSize) {
        //  math.abs(filterInfinity.last - localFilterInfinity.last).sign
        // } else {
        //  0.0
        // }
        // val accumulatedError = rep(0.0)(acc => acc + error)
        node.put(Sensors.accumulatedReward, accumulatedReward)
        node.put(Sensors.reward, reward)
        node.put(Sensors.fieldComputation, fieldComputation)
        node.put(Sensors.fieldSensing, fieldSensing)
        node.put(Sensors.localComputationWindow, windowComputation)
        node.put(Sensors.windowSensing, windowFieldSensing)
        node.put(Sensors.ticks, roundCounter())
        node.put(Sensors.error, error)
        // node.put(Sensors.accumulatedError, accumulatedError)
        node.put(Sensors.nextWakeUp, actionSpace(action)) // Actuation
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
      -weightForConvergence * (deltaFixed / actionSpace.max)
    } else {
      -(1 - (deltaFixed / actionSpace.max)) * (1 - weightForConvergence)
    }
  }

  def global(): CentralAgent[_, _] = alchemistEnvironment.getGlobalReactions
    .iterator()
    .asScala
    .collectFirst { case reaction: CentralAgent[_, _] => reaction }
    .get

  def deltaFixed: Double = alchemistDeltaTime(0.0)

}
