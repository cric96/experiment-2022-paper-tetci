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
    with BlockC
    with Gradients
    with FieldUtils
    with StateManagement {

  import Sensors._

  private lazy val localWindowSize = node.getOption(Sensors.window).getOrElse(3)
  private lazy val resultSize = 3
  private lazy val actionSpace = node.getOption(actions).getOrElse(List(1.0, 1.5, 2, 3))
  private lazy val sharedMemory: ReplayBuffer = loadMemory()
  private lazy val weightForConvergence = node.getOption(Sensors.weight).getOrElse(0.9)
  private lazy val configuredProgram: String =
    if (node.has("program")) {
      node.get[Any]("program") match {
        case n: Number =>
          n.intValue() match {
            case 1 => "hopGradient"
            case 2 => "broadcast"
            case 3 => "collect"
            case _ => "gradient"
          }
        case s: String => s
        case _ => "gradient"
      }
    } else "gradient"
  def activeProgram: String = global().activeProgramName(configuredProgram)
  def policy: (AgentState => Double) = loadPolicy()

  override def main(): Any = {
    val localComputation = computation()
    val computationWindow = rep(Queue.empty[Double])(queue => (queue :+ localComputation).takeRight(resultSize))
    node.put(Sensors.localComputation, localComputation)
    update(localComputation, computationWindow)
  }

  def update(localComputation: Double, elements: Queue[Double]): Unit = {
    val localSensing = perception()
    val (_, Some(action)) = rep((Option.empty[AgentState], Option.empty[Double])) { case (oldState, oldAction) =>
      val fieldComputation =
        includingSelf.reifyField(NeighborInfo(nbr(localComputation), nbrRange(), nbr(oldAction).getOrElse(1.0)))
      val fieldSensing =
        includingSelf.reifyField(NeighborInfo(nbr(localSensing), nbrRange(), nbr(oldAction).getOrElse(1.0)))

      val windowComputation = window(fieldComputation)
      val windowFieldSensing = window(fieldSensing)

      val state = new AgentState(mid(), windowComputation, windowFieldSensing)
      val reward = oldState.map(evalReward).getOrElse(0.0)
      val (action) = policy(state)
      oldState.zip(oldAction).foreach { case (stateT, actionT) =>
        val index = if (actionSpace.contains(actionT)) actionSpace.indexOf(actionT) else -1
        sharedMemory.put(stateT, index, actionT, reward, state)
      }
      val accumulatedReward = rep(0.0)(acc => acc + reward)
      val unstableTime = rep((0.0, localComputation)) { case (stable, old) =>
        (
          if (localComputation == old) { 0.0 }
          else { deltaFixed },
          localComputation
        )
      }._1
      val accumulatedUnstableTime = rep(0.0)(acc => acc + unstableTime)
      node.put("accumulatedUnstableTime", accumulatedUnstableTime)
      val allExperience = rep(Queue.empty[(AgentState, Int, Double, Double)]) { queue =>
        oldState.zip(oldAction) match {
          case Some((stateT, actionT)) =>
            val index = if (actionSpace.contains(actionT)) actionSpace.indexOf(actionT) else -1
            queue :+ (stateT, index, actionT, reward)
          case None =>
            queue
        }
      }
      val prog = activeProgram
      val groundTruth = global().groundTruthGradient(mid(), prog)
      val fullScale = global().programScale(prog)
      val absError =
        if (localComputation.isFinite && groundTruth.isFinite) math.abs(localComputation - groundTruth)
        else if (!localComputation.isFinite && !groundTruth.isFinite) 0.0
        else fullScale
      val normalizedStepError = math.min(1.0, absError / fullScale)
      val stepErrorIntegral = normalizedStepError * deltaFixed
      val accumulatedError = rep(0.0)(acc => acc + stepErrorIntegral)
      node.put(Sensors.groundTruth, groundTruth)
      node.put(Sensors.error, absError)
      node.put(Sensors.accumulatedError, accumulatedError)
      node.put("experiences", allExperience)
      node.put(Sensors.accumulatedReward, accumulatedReward)
      node.put(Sensors.reward, reward)
      node.put(Sensors.fieldComputation, fieldComputation)
      node.put(Sensors.fieldSensing, fieldSensing)
      node.put(Sensors.localComputationWindow, windowComputation)
      node.put(Sensors.windowSensing, windowFieldSensing)
      node.put(Sensors.ticks, roundCounter())
      node.put(Sensors.nextWakeUp, action) // Actuation
      (Some(state), Some(action))
    }
  }

  def exec(): Unit =
    node.put(Sensors.nextWakeUp, actionSpace.min)

  def computation(): Double = {
    val src = sense[Boolean](Sensors.source)
    val prog = activeProgram
    // Separate ScaFi alignment branches per building block so switching at episode boundaries is 100% clean
    branch(prog == "hopGradient") {
      hopDistance(src)
    } {
      branch(prog == "broadcast") {
        val g = classicGradient(src)
        val payload = if (src) (mid() + 1).toDouble * 10.0 else 0.0
        rep(payload) { value =>
          mux(g == 0.0)(payload)(
            excludingSelf.minHoodSelector[Double, Double](nbr(g) + nbrRange())(nbr(value)).getOrElse(payload)
          )
        }
      } {
        branch(prog == "collect") {
          val pot = classicGradient(src)
          C[Double, Double](pot, _ + _, 1.0, 0.0)
        } {
          classicGradient(src)
        }
      }
    }
  }
  def perception(): Double = 0.0 // what I need from my neighborhood for computing the state

  def window(snapshot: Map[ID, NeighborInfo]): Queue[Map[ID, NeighborInfo]] =
    rep(Queue.empty[Map[ID, NeighborInfo]])(queue => (queue :+ snapshot).takeRight(localWindowSize))

  def loadPolicy(): (AgentState => Double) = global().policy

  def loadMemory(): ReplayBuffer = global().memory

  def evalReward(state: AgentState): Double = {
    val myOutput = state.neighborhoodOutput.flatMap(_.get(state.me)).takeRight(4)
    val history = TemporalInfo.computeDeltaTrend(myOutput.map(_.data))
    node.put(Sensors.history, history)
    val rawIntensity = it.unibo.learning.network.NeuralNetworkRL.SpatialTemporal.instabilityIntensity(state)
    val effectiveIntensity = 0.06 + 0.94 * rawIntensity
    val minAct = 0.5
    val maxAct = actionSpace.max
    val u = math.max(0.0, math.min(1.0, (deltaFixed - minAct) / (maxAct - minAct)))
    val energyWeight = math.max(0.005, 1.0 - weightForConvergence)
    -weightForConvergence * effectiveIntensity * (u * u) - energyWeight * ((1.0 - u) * (1.0 - u))
  }

  def global(): CentralAgent[_, _] = alchemistEnvironment.getGlobalReactions
    .iterator()
    .asScala
    .collectFirst { case reaction: CentralAgent[_, _] => reaction }
    .get

  def deltaFixed: Double = alchemistDeltaTime(0.0)

}
