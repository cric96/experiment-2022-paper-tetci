package it.unibo.learning.network

import ai.djl.Model
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.nn.{AbstractBlock, Activation, Block, Parameter, SequentialBlock}
import ai.djl.nn.core.Linear
import ai.djl.training.{DefaultTrainingConfig, ParameterStore, Trainer}
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.PairList

import scala.jdk.CollectionConverters._
import scala.util.Using

/** Idiomatic Scala wrapper around DJL (PyTorch engine) for Deep Reinforcement Learning. */
final class TorchNetwork(
    val name: String,
    val inputDim: Int,
    val block: Block,
    val lr: Float = 1e-3f,
    val clipGrad: Float = 10.0f,
    val seed: Int = 42
) extends AutoCloseable {

  val model: Model = TorchNetwork.lock.synchronized {
    val m = Model.newInstance(name)
    m.setBlock(block)
    m
  }

  val trainer: Trainer = TorchNetwork.lock.synchronized {
    ai.djl.engine.Engine.getInstance().setRandomSeed(seed)
    val optimizer = Optimizer
      .adam()
      .optLearningRateTracker(Tracker.fixed(lr))
      .optClipGrad(clipGrad)
      .build()
    val config = new DefaultTrainingConfig(Loss.l2Loss())
      .optOptimizer(optimizer)
      .optInitializer(new XavierInitializer(), Parameter.Type.WEIGHT)
    val t = model.newTrainer(config)
    t.initialize(new Shape(1, inputDim))
    t
  }

  def manager: NDManager = trainer.getManager

  def forward(input: NDArray, training: Boolean = false): NDArray = TorchNetwork.lock.synchronized {
    if (training) trainer.forward(new NDList(input)).singletonOrThrow()
    else trainer.evaluate(new NDList(input)).singletonOrThrow()
  }

  def evaluate(input: NDArray): NDArray = TorchNetwork.lock.synchronized {
    trainer.evaluate(new NDList(input)).singletonOrThrow()
  }

  /** Extracts a lock-free fast inference function for standard 3-Linear MLP blocks (Linear -> ReLU -> Linear -> ReLU -> Linear [-> Tanh]). */
  def snapshotFastMLP(useTanh: Boolean = false): Array[Float] => Array[Float] = TorchNetwork.lock.synchronized {
    def extractLinear(b: Block): (Array[Float], Array[Float], Int, Int) = {
      val params = b.getParameters
      val wArr = params.get("weight").getArray
      val sh = wArr.getShape
      val outF = sh.get(0).toInt
      val inF = sh.get(1).toInt
      val w = wArr.toFloatArray
      val biasParam = params.get("bias")
      val bias = if (biasParam != null) biasParam.getArray.toFloatArray else new Array[Float](outF)
      (w, bias, inF, outF)
    }
    def applyLinear(in: Array[Float], spec: (Array[Float], Array[Float], Int, Int), relu: Boolean): Array[Float] = {
      val (w, b, inF, outF) = spec
      val out = new Array[Float](outF)
      var o = 0
      while (o < outF) {
        var s = b(o)
        val row = o * inF
        var i = 0
        while (i < inF) {
          s += w(row + i) * in(i)
          i += 1
        }
        out(o) = if (relu && s < 0.0f) 0.0f else s
        o += 1
      }
      out
    }
    val ch = block.getChildren.values().asScala.toSeq
    val l1 = extractLinear(ch(0))
    val l2 = extractLinear(ch(2))
    val l3 = extractLinear(ch(4))
    (x: Array[Float]) => {
      val h1 = applyLinear(x, l1, relu = true)
      val h2 = applyLinear(h1, l2, relu = true)
      val out = applyLinear(h2, l3, relu = false)
      if (useTanh) {
        var i = 0
        while (i < out.length) {
          out(i) = math.tanh(out(i)).toFloat
          i += 1
        }
      }
      out
    }
  }

  /** Extracts the last hidden layer activation (penultimate representation before the output head). */
  def extractHiddenEmbedding(input: NDArray): NDArray = TorchNetwork.lock.synchronized {
    val children = block.getChildren.values().asScala.toSeq
    // Run through all blocks except the final Linear (+ optional output Activation) head
    val hiddenBlocks = children.take(4) // Linear1 -> ReLU1 -> Linear2 -> ReLU2
    val paramStore = new ai.djl.training.ParameterStore(manager, false)
    var current = new NDList(input)
    hiddenBlocks.foreach { b =>
      current = b.forward(paramStore, current, false)
    }
    current.singletonOrThrow()
  }

  def trainStep(lossFn: => NDArray): Unit = TorchNetwork.lock.synchronized {
    Using.resource(trainer.newGradientCollector()) { gc =>
      val loss = lossFn
      gc.backward(loss)
    }
    trainer.step()
  }

  def withFrozenParameters[A](body: => A): A = TorchNetwork.lock.synchronized {
    block.freezeParameters(true)
    try body
    finally block.freezeParameters(false)
  }

  def softUpdateFrom(source: TorchNetwork, tau: Float): Unit = TorchNetwork.lock.synchronized {
    Using.resource(manager.newSubManager()) { sub =>
      val targetParams = block.getParameters.values().asScala
      val sourceParams = source.block.getParameters.values().asScala
      targetParams.zip(sourceParams).foreach { case (targetParam, sourceParam) =>
        val tArr = targetParam.getArray
        val sArr = sourceParam.getArray.toDevice(tArr.getDevice, false)
        val updated = sArr.mul(tau).add(tArr.mul(1.0f - tau))
        updated.copyTo(tArr)
        updated.attach(sub)
      }
    }
  }

  override def close(): Unit = {
    trainer.close()
    model.close()
  }
}

object TorchNetwork {
  private val lock = new AnyRef

  implicit final class RichNDArray(private val self: NDArray) extends AnyVal {
    def +(other: NDArray): NDArray = self.add(other)
    def +(scalar: Number): NDArray = self.add(scalar)
    def -(other: NDArray): NDArray = self.sub(other)
    def -(scalar: Number): NDArray = self.sub(scalar)
    def *(other: NDArray): NDArray = self.mul(other)
    def *(scalar: Number): NDArray = self.mul(scalar)
    def /(other: NDArray): NDArray = self.div(other)
    def /(scalar: Number): NDArray = self.div(scalar)
    def unary_- : NDArray = self.neg()
  }

  def withSubManager[A](parent: NDManager)(body: NDManager => A): A =
    TorchNetwork.lock.synchronized {
      Using.resource(parent.newSubManager())(body)
    }

  def mlpBlock(
      hiddenSizes: Seq[Int],
      outputSize: Int,
      outputActivation: Option[Block] = None
  ): SequentialBlock = {
    val layers: Seq[Block] =
      hiddenSizes.flatMap(h => Seq(Linear.builder().setUnits(h).build(), Activation.reluBlock())) :+
        Linear.builder().setUnits(outputSize).build()
    val allLayers = outputActivation.fold(layers)(layers :+ _)
    allLayers.foldLeft(new SequentialBlock())(_.add(_))
  }

  def createMLP(
      name: String,
      inputDim: Int,
      hiddenSizes: Seq[Int],
      outputSize: Int,
      outputActivation: Option[Block] = None,
      lr: Float = 1e-3f,
      seed: Int = 42
  ): TorchNetwork =
    new TorchNetwork(
      name = name,
      inputDim = inputDim,
      block = mlpBlock(hiddenSizes, outputSize, outputActivation),
      lr = lr,
      seed = seed
    )

  /** Edge-Conditioned Graph Attention + Relational Max-Pooling Network (EdgeGAT) on local star-graphs. */
  final class EdgeGATBlock(
      val selfDim: Int,
      val maxNeighbors: Int,
      val neighDim: Int,
      val edgeHidden: Int,
      val readoutHidden: Int,
      val outputSize: Int,
      val outputActivation: Option[Block] = None
  ) extends AbstractBlock {

    private val edgeMLP: SequentialBlock = addChildBlock(
      "edgeMLP",
      new SequentialBlock()
        .add(Linear.builder().setUnits(edgeHidden).build())
        .add(Activation.reluBlock())
        .add(Linear.builder().setUnits(edgeHidden).build())
        .add(Activation.reluBlock())
    )

    private val attnScorer: Linear = addChildBlock(
      "attnScorer",
      Linear.builder().setUnits(1).build()
    )

    private val readoutMLP: SequentialBlock = {
      val seq = new SequentialBlock()
        .add(Linear.builder().setUnits(readoutHidden).build())
        .add(Activation.reluBlock())
        .add(Linear.builder().setUnits(outputSize).build())
      outputActivation.foreach(seq.add)
      addChildBlock("readoutMLP", seq)
    }

    override protected def initializeChildBlocks(
        manager: NDManager,
        dataType: DataType,
        inputShapes: Shape*
    ): Unit = {
      edgeMLP.initialize(manager, dataType, new Shape(1, selfDim + neighDim))
      attnScorer.initialize(manager, dataType, new Shape(1, edgeHidden))
      readoutMLP.initialize(manager, dataType, new Shape(1, selfDim + 2 * edgeHidden))
    }

    override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = {
      val batchSize = inputShapes(0).get(0)
      Array(new Shape(batchSize, outputSize))
    }

    override protected def forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList[String, Object]
    ): NDList = {
      val x = inputs.singletonOrThrow()
      val batchSize = x.getShape.get(0)

      val selfFeat = x.get(new NDIndex(":, :{}", java.lang.Long.valueOf(selfDim)))
      val neighFlat = x.get(new NDIndex(":, {}:", java.lang.Long.valueOf(selfDim)))
      val neighTensor = neighFlat.reshape(batchSize, maxNeighbors, neighDim)

      val selfExpanded = selfFeat.reshape(batchSize, 1, selfDim).repeat(1, maxNeighbors)
      val pairInput = selfExpanded.concat(neighTensor, 2).reshape(batchSize * maxNeighbors, selfDim + neighDim)

      val edgeMessagesFlat = edgeMLP.forward(parameterStore, new NDList(pairInput), training, params).singletonOrThrow()
      val edgeMessages = edgeMessagesFlat.reshape(batchSize, maxNeighbors, edgeHidden)

      val attnLogits = attnScorer
        .forward(parameterStore, new NDList(edgeMessagesFlat), training, params)
        .singletonOrThrow()
        .reshape(batchSize, maxNeighbors, 1)
      val attnWeights = attnLogits.softmax(1)

      // Combine Graph Attention Pooling + Relational Max-Pooling (EdgeConv)
      val attnPooled = edgeMessages.mul(attnWeights).sum(Array(1))
      val maxPooled = edgeMessages.max(Array(1))

      val combined = selfFeat.concat(attnPooled, 1).concat(maxPooled, 1)
      readoutMLP.forward(parameterStore, new NDList(combined), training, params)
    }

    /** Extracts a lock-free fast inference function from the current PyTorch parameters for parallel simulation rounds. */
    def snapshotFastInference(): Array[Float] => Float = TorchNetwork.lock.synchronized {
      def extractLinear(b: Block): (Array[Float], Array[Float], Int, Int) = {
        val params = b.getParameters
        val wArr = params.get("weight").getArray
        val sh = wArr.getShape
        val outF = sh.get(0).toInt
        val inF = sh.get(1).toInt
        val w = wArr.toFloatArray
        val biasParam = params.get("bias")
        val bias = if (biasParam != null) biasParam.getArray.toFloatArray else new Array[Float](outF)
        (w, bias, inF, outF)
      }
      def applyLinear(in: Array[Float], spec: (Array[Float], Array[Float], Int, Int), relu: Boolean): Array[Float] = {
        val (w, b, inF, outF) = spec
        val out = new Array[Float](outF)
        var o = 0
        while (o < outF) {
          var s = b(o)
          val row = o * inF
          var i = 0
          while (i < inF) {
            s += w(row + i) * in(i)
            i += 1
          }
          out(o) = if (relu && s < 0.0f) 0.0f else s
          o += 1
        }
        out
      }
      val edgeCh = edgeMLP.getChildren.values().asScala.toSeq
      val eL1 = extractLinear(edgeCh(0))
      val eL2 = extractLinear(edgeCh(2))
      val attL = extractLinear(attnScorer)
      val readCh = readoutMLP.getChildren.values().asScala.toSeq
      val rL1 = extractLinear(readCh(0))
      val rL2 = extractLinear(readCh(2))
      val useTanh = outputActivation.isDefined

      (x: Array[Float]) => {
        val selfF = java.util.Arrays.copyOfRange(x, 0, selfDim)
        val msgs = new Array[Array[Float]](maxNeighbors)
        val logits = new Array[Float](maxNeighbors)
        var maxLogit = Float.NegativeInfinity
        var k = 0
        while (k < maxNeighbors) {
          val pair = new Array[Float](selfDim + neighDim)
          System.arraycopy(selfF, 0, pair, 0, selfDim)
          System.arraycopy(x, selfDim + k * neighDim, pair, selfDim, neighDim)
          val h1 = applyLinear(pair, eL1, relu = true)
          val m = applyLinear(h1, eL2, relu = true)
          msgs(k) = m
          val l = applyLinear(m, attL, relu = false)(0)
          logits(k) = l
          if (l > maxLogit) maxLogit = l
          k += 1
        }
        val alphas = new Array[Float](maxNeighbors)
        var sumExp = 0.0f
        k = 0
        while (k < maxNeighbors) {
          val e = math.exp(logits(k) - maxLogit).toFloat
          alphas(k) = e
          sumExp += e
          k += 1
        }
        val attPool = new Array[Float](edgeHidden)
        val maxPool = Array.fill(edgeHidden)(Float.NegativeInfinity)
        k = 0
        while (k < maxNeighbors) {
          val a = alphas(k) / sumExp
          val m = msgs(k)
          var d = 0
          while (d < edgeHidden) {
            attPool(d) += a * m(d)
            if (m(d) > maxPool(d)) maxPool(d) = m(d)
            d += 1
          }
          k += 1
        }
        val comb = new Array[Float](selfDim + 2 * edgeHidden)
        System.arraycopy(selfF, 0, comb, 0, selfDim)
        System.arraycopy(attPool, 0, comb, selfDim, edgeHidden)
        System.arraycopy(maxPool, 0, comb, selfDim + edgeHidden, edgeHidden)
        val r1 = applyLinear(comb, rL1, relu = true)
        val r2 = applyLinear(r1, rL2, relu = false)(0)
        if (useTanh) math.tanh(r2).toFloat else r2
      }
    }

    /** Extracts both the 48D readout hidden embedding h_i(t) and the K=8 neighbor attention weights alpha_ij(t). */
    def snapshotEmbeddingAndAttention(): Array[Float] => (Array[Float], Array[Float]) = TorchNetwork.lock.synchronized {
      def extractLinear(b: Block): (Array[Float], Array[Float], Int, Int) = {
        val params = b.getParameters
        val wArr = params.get("weight").getArray
        val sh = wArr.getShape
        val outF = sh.get(0).toInt
        val inF = sh.get(1).toInt
        val w = wArr.toFloatArray
        val biasParam = params.get("bias")
        val bias = if (biasParam != null) biasParam.getArray.toFloatArray else new Array[Float](outF)
        (w, bias, inF, outF)
      }
      def applyLinear(in: Array[Float], spec: (Array[Float], Array[Float], Int, Int), relu: Boolean): Array[Float] = {
        val (w, b, inF, outF) = spec
        val out = new Array[Float](outF)
        var o = 0
        while (o < outF) {
          var s = b(o)
          val row = o * inF
          var i = 0
          while (i < inF) {
            s += w(row + i) * in(i)
            i += 1
          }
          out(o) = if (relu && s < 0.0f) 0.0f else s
          o += 1
        }
        out
      }
      val edgeCh = edgeMLP.getChildren.values().asScala.toSeq
      val eL1 = extractLinear(edgeCh(0))
      val eL2 = extractLinear(edgeCh(2))
      val attL = extractLinear(attnScorer)
      val readCh = readoutMLP.getChildren.values().asScala.toSeq
      val rL1 = extractLinear(readCh(0))

      (x: Array[Float]) => {
        val selfF = java.util.Arrays.copyOfRange(x, 0, selfDim)
        val msgs = new Array[Array[Float]](maxNeighbors)
        val logits = new Array[Float](maxNeighbors)
        var maxLogit = Float.NegativeInfinity
        var k = 0
        while (k < maxNeighbors) {
          val pair = new Array[Float](selfDim + neighDim)
          System.arraycopy(selfF, 0, pair, 0, selfDim)
          System.arraycopy(x, selfDim + k * neighDim, pair, selfDim, neighDim)
          val h1 = applyLinear(pair, eL1, relu = true)
          val m = applyLinear(h1, eL2, relu = true)
          msgs(k) = m
          val l = applyLinear(m, attL, relu = false)(0)
          logits(k) = l
          if (l > maxLogit) maxLogit = l
          k += 1
        }
        val alphas = new Array[Float](maxNeighbors)
        var sumExp = 0.0f
        k = 0
        while (k < maxNeighbors) {
          val e = math.exp(logits(k) - maxLogit).toFloat
          alphas(k) = e
          sumExp += e
          k += 1
        }
        val attPool = new Array[Float](edgeHidden)
        val maxPool = Array.fill(edgeHidden)(Float.NegativeInfinity)
        k = 0
        while (k < maxNeighbors) {
          alphas(k) = alphas(k) / sumExp
          val a = alphas(k)
          val m = msgs(k)
          var d = 0
          while (d < edgeHidden) {
            attPool(d) += a * m(d)
            if (m(d) > maxPool(d)) maxPool(d) = m(d)
            d += 1
          }
          k += 1
        }
        val comb = new Array[Float](selfDim + 2 * edgeHidden)
        System.arraycopy(selfF, 0, comb, 0, selfDim)
        System.arraycopy(attPool, 0, comb, selfDim, edgeHidden)
        System.arraycopy(maxPool, 0, comb, selfDim + edgeHidden, edgeHidden)
        val hiddenReadout = applyLinear(comb, rL1, relu = true)
        (hiddenReadout, alphas)
      }
    }
  }

  def createEdgeGAT(
      name: String,
      selfDim: Int,
      maxNeighbors: Int,
      neighDim: Int,
      edgeHidden: Int = 32,
      readoutHidden: Int = 64,
      outputSize: Int = 1,
      outputActivation: Option[Block] = None,
      lr: Float = 1e-3f,
      seed: Int = 42
  ): TorchNetwork =
    new TorchNetwork(
      name = name,
      inputDim = selfDim + maxNeighbors * neighDim,
      block = new EdgeGATBlock(selfDim, maxNeighbors, neighDim, edgeHidden, readoutHidden, outputSize, outputActivation),
      lr = lr,
      seed = seed
    )

  /** Smooth L1 (Huber) loss: 0.5 * x^2 if |x| < 1 else |x| - 0.5 */
  def smoothL1Loss(predicted: NDArray, target: NDArray): NDArray = {
    val diff = predicted.sub(target).abs()
    val quadratic = diff.minimum(1.0f)
    val linear = diff.sub(quadratic)
    quadratic.pow(2).mul(0.5f).add(linear).mean()
  }

  def l2Normalize(vec: Array[Double]): Array[Float] = {
    val norm = math.sqrt(vec.foldLeft(0.0)((acc, x) => acc + x * x))
    if (norm < 1e-12) vec.map(_.toFloat) else vec.map(x => (x / norm).toFloat)
  }

  final case class MinMaxActionScaler(min: Double, max: Double) {
    require(max > min, s"max ($max) must be greater than min ($min)")
    private val halfRange = (max - min) / 2.0
    private val mid = (max + min) / 2.0

    def scale(action: Double): Float =
      ((math.max(min, math.min(max, action)) - mid) / halfRange).toFloat

    def unscale(scaledAction: Double): Double =
      mid + math.max(-1.0, math.min(1.0, scaledAction)) * halfRange
  }
}
