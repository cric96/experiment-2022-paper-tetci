package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.abstractions.{AgentState, Contextual}
import it.unibo.learning.network.NeuralNetworkRL.{Historical, Spatial}
import it.unibo.learning.network.torch.{d3rlpy, numpy}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.PyQuote

import scala.collection.immutable.Queue
import scala.util.Random

class OfflineLearner(actionSpace: Seq[Double], randomKey: Double, weight: Double) extends Learner {

  private val encoder: AgentState => py.Any =
    // Spatial.encodeSpatial(_, 5, false)
    Historical.encodeHistory(_, 20)

  val steps = 10000
  val explorationSteps = 10
  val learningStepAfter = 20
  val random = new Random(42)
  val epsilon = 0.99
  // private val observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()
  private val algorithm =
    d3rlpy.algos.DiscreteBCQConfig(gamma = 0.9).create("cuda:0")
  private val explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(0.99, 0.0, explorationSteps)
  algorithm.create_impl(py"[20]", py"${actionSpace.size}")
  var ticks = 0
  def policy: (AgentState => Double) = { state =>
    val encoded = numpy.array(encoder(state)).reshape(1, -1)
    val action = if (ticks < explorationSteps) {
      explorer.sample(algorithm, encoded, ticks)
    } else {
      algorithm.predict(encoded)
    }

    actionSpace(action.bracketAccess(0).as[Int])
  }

  def store(where: String): Unit = {} // store optimal policy
  def load(where: String): (AgentState => Double) = null // load policy stored
  def update(batch: Seq[Experience]): Unit = {} // update the internal state following the experience computed
  def injectRandom(random: Random): Unit = {}
  def injectCentralAgent(agent: CentralAgent[_, _]): Unit = {}

  var lastBatch: Seq[Seq[(AgentState, Int, Double, Double)]] = Seq.empty
  override def endEpisode(replyBuffer: Seq[Seq[(AgentState, Int, Double, Double)]]): Unit = {
    lastBatch = lastBatch ++ replyBuffer
    if (ticks < (explorationSteps + 1)) { // n of exploration, one of exploit
      val observations = lastBatch.flatMap(_.map(_._1).map(encoder))
      val actions = lastBatch.flatMap(_.map(_._2))
      println(actions.size)
      val actionsDoubles = lastBatch.flatMap(_.map(_._3))
      val rewards = lastBatch.flatMap(_.map(_._4))
      val terminalsZero = lastBatch.flatMap(list => Queue.fill(list.size)(0.0))
      val terminals = lastBatch.flatMap(list => Queue.fill(list.size - 1)(0.0) :+ 1.0)
      val npObservations = numpy.array(observations.toPythonCopy)
      val npActions = numpy.array(actions.map(_.as[py.Any]).toPythonCopy)
      val npRewards = numpy.array(rewards.map(_.as[py.Any]).toPythonCopy)
      val npTerminalsZero = numpy.array(terminalsZero.map(_.as[py.Any]).toPythonCopy)
      val npTerminals = numpy.array(terminals.map(_.as[py.Any]).toPythonCopy)
      val dataset = d3rlpy.dataset.MDPDataset(npObservations, npActions, npRewards, npTerminalsZero, npTerminals)
      val evaluator = d3rlpy.metrics.TDErrorEvaluator(episodes = dataset.episodes)
      val nSteps = if (ticks < explorationSteps) steps else steps * learningStepAfter
      algorithm.fit(
        dataset,
        n_steps = nSteps,
        evaluators = py"{'td_error': $evaluator}",
        experiment_name = s"dqn-$randomKey-$weight-$ticks"
      )
      dataset.del()
    }
    ticks += 1
  }
}
