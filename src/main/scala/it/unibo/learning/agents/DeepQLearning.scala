package it.unibo.learning.agents
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.{AgentState, Contextual, DecayReference, ReplayBuffer}
import it.unibo.learning.network.NeuralNetworkRL
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

import scala.util.Random

class DeepQLearning(
    epsilon: DecayReference[Double],
    alpha: Double,
    gamma: Double,
    copyEach: Int,
    referenceNet: NeuralNetworkRL
) extends Learner {
  private val device = torch.device("cuda:0")
  private val gc = py.module("gc")
  var updates = 0
  private var random: Random = new Random()
  private val targetNetwork = referenceNet.cloneNetwork
  private val policyNetwork = referenceNet
  targetNetwork.underlying.to(device)
  policyNetwork.underlying.to(device)
  private val optimizer = optim.RMSprop(policyNetwork.underlying.parameters(), alpha)
  private val behaviouralPolicy = policyNetwork.policy(device)

  override def policy: AgentState => (Int, Contextual) = state =>
    if (random.nextDouble() < epsilon) {
      (random.shuffle(policyNetwork.actionSpace.indices.toList).head, policyNetwork.emptyContextual)
    } else behaviouralPolicy(state)

  override def store(where: String): Unit = {}

  override def load(where: String): (AgentState, (Int, Contextual)) = null

  override def update(batch: Seq[ReplayBuffer.Experience]): Unit = {
    val session = PythonMemoryManager.session()
    // context
    import session._
    val states = batch.map(_.stateT).map(referenceNet.encode)
    val action = batch.map(_.actionT).map(action => action).toPythonCopy
    val rewards = torch.tensor(batch.map(_.rewardTPlus).toPythonCopy, device = device).record()
    val actionSelection = torch
      .tensor(action, device = device)
      .record()
      .view(batch.size, 1)
      .record()
    val nextStates = batch.map(_.stateTPlus).map(referenceNet.encode)
    val tensorInputStates = referenceNet.encodeBatch(states, device).record()
    val stateActionValue =
      policyNetwork
        .underlying(tensorInputStates)
        .record()
        .gather(1, actionSelection)
        .record()
    val nextStateValues =
      targetNetwork
        .underlying(referenceNet.encodeBatch(nextStates, device).record())
        .record()
        .max(1)
        .record()
        .bracketAccess(0)
        .record()
        .detach()
        .record()
    val expectedValue = ((nextStateValues * gamma).record() + rewards).record()
    val criterion = nn.SmoothL1Loss()
    val loss = criterion(stateActionValue, expectedValue.unsqueeze(1).record()).record()
    optimizer.zero_grad()
    loss.backward().record()
    writer.add_scalar("Loss", loss.detach().item().as[Double], updates)
    torch.nn.utils.clip_grad_value_(policyNetwork.underlying.parameters(), 1)
    optimizer.step()
    session.clear()
    updates += 1
    if (updates % copyEach == 0) {
      targetNetwork.underlying.load_state_dict(policyNetwork.underlying.state_dict())
    }
  }

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: CentralAgent[_, _]): Unit = agent.attachDecayable("epsilon" -> epsilon)
}
