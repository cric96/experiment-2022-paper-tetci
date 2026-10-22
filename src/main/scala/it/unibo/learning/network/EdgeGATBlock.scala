package it.unibo.learning.network

import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.nn.core.Linear
import ai.djl.nn.{AbstractBlock, Activation, Block, SequentialBlock}
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import it.unibo.learning.network.FastInference.DenseLayer

/** Edge-conditioned graph attention with relational max-pooling (EdgeGAT) over a star graph, encoded as the ego
  * features followed by `maxNeighbors` slots of `neighDim` features (see [[it.unibo.learning.encoding.StarGraph]]).
  *
  * Each (ego, neighbour) pair is turned into a message by an edge MLP; messages are pooled both with attention weights
  * (softmax of a linear score) and with an element-wise max, and the readout MLP maps ego features and pooled messages
  * to the output.
  */
final class EdgeGATBlock(
    val selfDim: Int,
    val maxNeighbors: Int,
    val neighDim: Int,
    val edgeHidden: Int,
    val readoutHidden: Int,
    val outputSize: Int,
    val outputActivation: Option[Block] = None
) extends AbstractBlock {
  // The children are registered (and initialized) in this order.
  private val edgeMLP: SequentialBlock = addChildBlock(
    "edgeMLP",
    new SequentialBlock()
      .add(Linear.builder().setUnits(edgeHidden.toLong).build())
      .add(Activation.reluBlock())
      .add(Linear.builder().setUnits(edgeHidden.toLong).build())
      .add(Activation.reluBlock())
  )

  private val attentionScorer: Linear = addChildBlock("attnScorer", Linear.builder().setUnits(1).build())

  private val readoutMLP: SequentialBlock = {
    val readout = new SequentialBlock()
      .add(Linear.builder().setUnits(readoutHidden.toLong).build())
      .add(Activation.reluBlock())
      .add(Linear.builder().setUnits(outputSize.toLong).build())
    outputActivation.foreach(readout.add)
    addChildBlock("readoutMLP", readout)
  }

  private val pairSize: Int = selfDim + neighDim

  override protected def initializeChildBlocks(manager: NDManager, dataType: DataType, inputShapes: Shape*): Unit = {
    edgeMLP.initialize(manager, dataType, new Shape(1, pairSize.toLong))
    attentionScorer.initialize(manager, dataType, new Shape(1, edgeHidden.toLong))
    readoutMLP.initialize(manager, dataType, new Shape(1, (selfDim + 2 * edgeHidden).toLong))
  }

  override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] =
    Array(new Shape(inputShapes(0).get(0), outputSize.toLong))

  override protected def forwardInternal(
      parameterStore: ParameterStore,
      inputs: NDList,
      training: Boolean,
      params: PairList[String, Object]
  ): NDList = {
    def run(block: Block, input: NDArray): NDArray =
      block.forward(parameterStore, new NDList(input), training, params).singletonOrThrow()
    val graphs = inputs.singletonOrThrow()
    val batchSize = graphs.getShape.get(0)
    val ego = graphs.get(new NDIndex(":, :{}", java.lang.Long.valueOf(selfDim.toLong)))
    val flatMessages = run(edgeMLP, pairsOf(graphs, ego, batchSize))
    val messages = flatMessages.reshape(batchSize, maxNeighbors.toLong, edgeHidden.toLong)
    val attention = run(attentionScorer, flatMessages).reshape(batchSize, maxNeighbors.toLong, 1).softmax(1)
    val pooled = ego.concat(messages.mul(attention).sum(Array(1)), 1).concat(messages.max(Array(1)), 1)
    readoutMLP.forward(parameterStore, new NDList(pooled), training, params)
  }

  /** One (ego, neighbour) row per graph and neighbour slot. */
  private def pairsOf(graphs: NDArray, ego: NDArray, batchSize: Long): NDArray = {
    val neighbours = graphs
      .get(new NDIndex(":, {}:", java.lang.Long.valueOf(selfDim.toLong)))
      .reshape(batchSize, maxNeighbors.toLong, neighDim.toLong)
    val repeatedEgo = ego.reshape(batchSize, 1, selfDim.toLong).repeat(1, maxNeighbors.toLong)
    repeatedEgo.concat(neighbours, 2).reshape(batchSize * maxNeighbors, pairSize.toLong)
  }

  /** A lock-free policy evaluation on the current parameters (single output). */
  def snapshotFastInference(): Array[Float] => Float = {
    val snapshot = fastSnapshot()
    snapshot.output
  }

  /** A lock-free function returning the readout embedding and the attention over the neighbour slots. */
  def snapshotEmbeddingAndAttention(): Array[Float] => (Array[Float], Array[Float]) = {
    val snapshot = fastSnapshot()
    snapshot.embeddingAndAttention
  }

  private def fastSnapshot(): FastEdgeGAT = TorchNetwork.serialized {
    FastEdgeGAT(
      FastInference.denseLayersOf(edgeMLP), DenseLayer.of(attentionScorer), FastInference.denseLayersOf(readoutMLP),
      selfDim, maxNeighbors, neighDim, tanhOutput = outputActivation.isDefined
    )
  }
}

/** The forward pass of an [[EdgeGATBlock]] on a snapshot of its parameters (see [[FastInference]]). */
final case class FastEdgeGAT(
    edgeLayers: Seq[DenseLayer],
    scorer: DenseLayer,
    readoutLayers: Seq[DenseLayer],
    selfDim: Int,
    maxNeighbors: Int,
    neighDim: Int,
    tanhOutput: Boolean
) {

  /** The hidden layer of the readout and the attention weights over the neighbour slots. */
  def embeddingAndAttention(graph: Array[Float]): (Array[Float], Array[Float]) = {
    val ego = graph.slice(0, selfDim)
    val messages = (0 until maxNeighbors).map(slot => messageOf(ego, graph, slot))
    val attention = FastInference.softmax(messages.map(scorer(_, relu = false)(0)).toArray)
    val pooled = ego ++ attentionPooling(messages, attention) ++ maxPooling(messages)
    (readoutLayers.head(pooled, relu = true), attention)
  }

  def output(graph: Array[Float]): Float = {
    val (embedding, _) = embeddingAndAttention(graph)
    val value = readoutLayers.last(embedding, relu = false)(0)
    if tanhOutput then math.tanh(value.toDouble).toFloat else value
  }

  private def messageOf(ego: Array[Float], graph: Array[Float], slot: Int): Array[Float] = {
    val start = selfDim + slot * neighDim
    edgeLayers.foldLeft(ego ++ graph.slice(start, start + neighDim))((input, layer) => layer(input, relu = true))
  }

  private def attentionPooling(messages: Seq[Array[Float]], attention: Array[Float]): Array[Float] =
    Array.tabulate(messages.head.length) { unit =>
      messages.indices.foldLeft(0.0f)((sum, slot) => sum + attention(slot) * messages(slot)(unit))
    }

  private def maxPooling(messages: Seq[Array[Float]]): Array[Float] =
    Array.tabulate(messages.head.length) { unit =>
      messages.foldLeft(Float.NegativeInfinity)((max, message) => if message(unit) > max then message(unit) else max)
    }
}
