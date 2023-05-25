package it.unibo.learning.agents
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.learning.abstractions.{AgentState, Contextual, DecayReference, ReplayBuffer}
import it.unibo.learning.network.NeuralNetworkRL
import it.unibo.learning.network.torch._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.util.Random

class DeepQLearning(
    epsilon: DecayReference[Double],
    alpha: Double,
    gamma: Double,
    copyEach: Int,
    referenceNet: NeuralNetworkRL,
    deviceName: String = "cpu"
) extends Learner {
  private val device =
    torch.device(if (deviceName != "cpu" && torch.cuda.is_available().as[Boolean]) deviceName else "cpu")
  var updates = 0
  private var random: Random = new Random()
  private var targetNetwork = referenceNet.cloneNetwork
  private var policyNetwork = referenceNet
  targetNetwork.underlying.to(device)
  policyNetwork.underlying.to(device)
  private val optimizer = optim.RMSprop(policyNetwork.underlying.parameters(), alpha)
  private val behaviouralPolicy = policyNetwork.policy(device)
  targetNetwork.underlying.eval()
  policyNetwork.underlying.eval()

  override def policy: AgentState => (Int, Contextual) = state =>
    if (random.nextDouble() < epsilon) {
      (random.shuffle(policyNetwork.actionSpace.indices.toList).head, policyNetwork.emptyContextual)
    } else behaviouralPolicy(state)

  override def store(where: String): Unit = {}

  override def load(where: String): (AgentState, (Int, Contextual)) = null

  override def update(batch: Seq[ReplayBuffer.Experience]): Unit = {
    implicit val session: PythonMemoryManager.Session = PythonMemoryManager.session()
    // context
    import session._
    targetNetwork.underlying.train()
    policyNetwork.underlying.train()

    val states = batch.map(_.stateT).map(referenceNet.encode)
    val action = batch.map(_.actionT).map(action => action).toPythonCopy
    val rewards = torch.tensor(batch.map(_.rewardTPlus).toPythonCopy, device = device).record()
    val nextStates = batch.map(_.stateTPlus).map(referenceNet.encode)
    val inputBatch = referenceNet.encodeBatch(states, device).record()
    val stateActionValue =
      policyNetwork
        .forward(inputBatch)
        .record()
        .gather(
          1,
          torch
            .tensor(action, device = device)
            .record()
            .view(batch.size, 1)
            .record()
        )
        .record()
    val nextStateValues = py.`with`(torch.no_grad()) { _ =>
      policyNetwork
        .forward(referenceNet.encodeBatch(nextStates, device).record())
        .record()
        .max(1)
        .record()
        .bracketAccess(0)
        .record()
    }
    val expectedValue = ((nextStateValues * gamma).record() + rewards).record()
    val criterion = nn.SmoothL1Loss()
    val loss = criterion(stateActionValue, expectedValue.unsqueeze(1).record()).record()
    optimizer.zero_grad()
    loss.backward().record()
    writer.add_scalar("Loss", loss.detach().item().as[Double], updates)
    torch.nn.utils.clip_grad_value_(policyNetwork.underlying.parameters(), 1)
    optimizer.step()
    policyNetwork.underlying.zero_grad(set_to_none = true)
    targetNetwork.underlying.zero_grad(set_to_none = true)
    session.clear()
    updates += 1
    if (updates % copyEach == 0) {
      targetNetwork.underlying.load_state_dict(policyNetwork.underlying.state_dict())
    }
    targetNetwork.underlying.eval()
    policyNetwork.underlying.eval()
  }

  override def injectRandom(random: Random): Unit = this.random = random

  override def injectCentralAgent(agent: CentralAgent[_, _]): Unit = agent.attachDecayable("epsilon" -> epsilon)
}
