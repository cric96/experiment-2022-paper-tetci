package it.unibo.learning.agents

import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.network.NeuralNetworkRL.Historical
import it.unibo.learning.network.torch.{d3rlpy, numpy}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from
import me.shadaj.scalapy.py.PyQuote

import scala.collection.immutable.Queue
import scala.util.Random

class OfflineLearnerContinuous(min: Double, max: Double, randomKey: Double, weight: Double) extends Learner {

  private val encoder: AgentState => py.Any =
    // Spatial.encodeSpatial(_, 5, false)
    Historical.encodeHistory(_, 20)
  val steps = 10000
  val explorationSteps = 10
  val learningStepAfter = 20
  var random = new Random(42)
  val epsilon = 0.99
  def linearScaler(start: Double, end: Double, steps: Int, current: Int): Double =
    if (current >= steps) end
    else start + (end - start) * current / steps
  val actionScaler = d3rlpy.preprocessing.MinMaxActionScaler(minimum = min, maximum = max)
  private val algorithm = d3rlpy.algos.BEARConfig(gamma = 0.9, action_scaler = actionScaler).create("cuda:0")
  algorithm.create_impl(py"[20]", py"${1}")
  var ticks = 0
  def policy: (AgentState => Double) = { state =>
    val encoded = numpy.array(encoder(state)).reshape(1, -1)
    // min max
    val norm = numpy.linalg.norm(encoded)
    val normalizedEncoded = encoded / norm
    val choice = random.nextDouble()
    val action = if (choice < linearScaler(epsilon, 0.0, explorationSteps, ticks)) {
      (random.nextDouble() * 4) + 1
    } else {
      val action = algorithm.predict(encoded)
      action.bracketAccess(0).as[Double]
    }
    action
  }

  def store(where: String): Unit = {} // store optimal policy
  def load(where: String): (AgentState => Double) = null // load policy stored
  def update(batch: Seq[Experience]): Unit = {} // update the internal state following the experience computed
  def injectRandom(random: Random): Unit = this.random = random
  def injectCentralAgent(agent: CentralAgent[_, _]): Unit = {}

  var lastBatch: Seq[Seq[(AgentState, Int, Double, Double)]] = Seq.empty
  override def endEpisode(replyBuffer: Seq[Seq[(AgentState, Int, Double, Double)]]): Unit = {
    lastBatch = lastBatch ++ replyBuffer
    if (ticks < explorationSteps + 1) {
      val observations = lastBatch.flatMap(_.map(_._1).map(encoder))
      val actions = lastBatch.flatMap(_.map(_._2))
      val actionsDoubles = lastBatch.flatMap(_.map(_._3))
      val rewards = lastBatch.flatMap(_.map(_._4))
      val terminalsZero = lastBatch.flatMap(list => Queue.fill(list.size)(0.0))
      val terminals = lastBatch.flatMap(list => Queue.fill(list.size - 1)(0.0) :+ 1.0)
      val npObservations = numpy.array(observations.toPythonCopy)
      val npActionDoubles = numpy.array(actionsDoubles.map(_.as[py.Any]).toPythonCopy)
      val npRewards = numpy.array(rewards.map(_.as[py.Any]).toPythonCopy)
      val npTerminalsZero = numpy.array(terminalsZero.map(_.as[py.Any]).toPythonCopy)
      val npTerminals = numpy.array(terminals.map(_.as[py.Any]).toPythonCopy)
      val dataset = d3rlpy.dataset.MDPDataset(npObservations, npActionDoubles, npRewards, npTerminalsZero, npTerminals)
      val evaluator = d3rlpy.metrics.TDErrorEvaluator(episodes = dataset.episodes)
      val nSteps = if (ticks < explorationSteps) steps else steps * learningStepAfter
      algorithm.fit(dataset, n_steps = nSteps, evaluators = py"{'td_error': $evaluator}")
      dataset.del()
    }
    ticks += 1
  }
}
