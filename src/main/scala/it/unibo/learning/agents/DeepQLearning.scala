package it.unibo.learning.agents
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.{AgentState, Contextual, DecayReference, ReplayBuffer}
import it.unibo.learning.agents.DeepQLearning.stateEncoding
import it.unibo.learning.network.{DQN, torch}
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

import scala.util.Random

class DeepQLearning(
    actionSpace: List[Double],
    epsilon: DecayReference[Double],
    alpha: Double,
    gamma: Double,
    hiddenSize: Int,
    copyEach: Int,
    window: Int
) extends Learner {
  var updates = 0
  private var random: Random = new Random()
  private val targetNetwork = DQN(window, hiddenSize, actionSpace.size)
  private val policyNetwork = DQN(window, hiddenSize, actionSpace.size)
  private val optimizer = optim.RMSprop(policyNetwork.parameters(), alpha)
  private val behaviouralPolicy = DeepQLearning.policyFromNetwork(policyNetwork, window)

  override def policy: AgentState => (Int, Contextual) = state =>
    if (random.nextDouble() < epsilon) {
      (random.shuffle(actionSpace.indices.toList).head, ())
    } else (behaviouralPolicy(state), ())

  override def store(where: String): Unit = {}

  override def load(where: String): (AgentState, (Int, Contextual)) = null

  override def update(batch: Seq[ReplayBuffer.Experience]): Unit = {
    val states = batch.map(_.stateT).map(state => stateEncoding(state, window).toPythonCopy).toPythonCopy
    val action = batch.map(_.actionT).map(action => actionSpace.indexOf(action)).toPythonCopy
    val rewards = torch.tensor(batch.map(_.rewardTPlus).toPythonCopy)
    val nextState = batch.map(_.stateTPlus).map(state => stateEncoding(state, window).toPythonCopy).toPythonCopy
    val stateActionValue = policyNetwork(torch.tensor(states)).gather(1, torch.tensor(action).view(batch.size, 1))
    val nextStateValues = targetNetwork(torch.tensor(nextState)).max(1).bracketAccess(0).detach()
    val expectedValue = (nextStateValues * gamma) + rewards
    val criterion = nn.SmoothL1Loss()
    val loss = criterion(stateActionValue, expectedValue.unsqueeze(1))
    torch.writer.add_scalar("Loss", loss.item().as[Double], updates)
    optimizer.zero_grad()
    loss.backward()
    py"[param.grad.data.clamp_(-1, 1) for param in ${policyNetwork.parameters()}]"
    optimizer.step()
    updates += 1
    if (updates % copyEach == 0) {
      targetNetwork.load_state_dict(policyNetwork.state_dict())
    }
  }

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: CentralAgent[_, _]): Unit = agent.attachDecayable("epsilon" -> epsilon)
}

object DeepQLearning {
  def policyFromNetwork(network: py.Dynamic, window: Int): AgentState => Int = { state =>
    val netInput = stateEncoding(state, window)
    py.`with`(torch.no_grad()) { _ =>
      val tensor = torch.tensor(netInput.toPythonCopy).view(1, window)
      val actionIndex = network(tensor).max(1).bracketAccess(1).item().as[Int]
      actionIndex
    }
  }

  def stateEncoding(state: AgentState, window: Int): Seq[Double] = {
    val states: LazyList[Double] = state.neighborhoodOutput
      .map(_(state.me))
      .map(_.data)
      .map {
        case x if !x.isFinite => -1
        case x => x
      }
      .to(LazyList)
    val fill: LazyList[Double] = LazyList.continually(0.0)
    (states #::: fill).take(window)
  }
}
