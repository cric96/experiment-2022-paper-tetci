package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import it.unibo.learning.network.TorchNetwork
import it.unibo.learning.network.TorchNetwork._

import java.io.File

/** Post-training inspection of a continuous actor-critic policy, exported for the explainability figures. */
final class ActorInspection(
    actor: TorchNetwork,
    critic: TorchNetwork,
    actionScaler: TorchNetwork.MinMaxActionScaler,
    stateDim: Int,
    outputDirectory: File
) {
  private val rootManager: NDManager = NDManager.newBaseManager()

  def export(allSamples: Vector[Array[Float]], weight: Double): Unit = {
    val outDir = outputDirectory
    outDir.mkdirs()
    if (allSamples.isEmpty) return

    withSubManager(rootManager) { sub =>
      val statesArr = allSamples.toArray
      val statesTensor = sub.create(statesArr)
      val hiddenTensor = actor.extractHiddenEmbedding(statesTensor)
      val predScaled = actor.evaluate(statesTensor)
      val qTensor = critic.evaluate(statesTensor.concat(predScaled, 1))

      val nSamples = allSamples.size
      val hiddenDim = hiddenTensor.getShape.get(1).toInt

      // Compute finite-difference feature sensitivity |d(Delta t) / d(x_k)| and permutation impact for each feature k
      val eps = 0.1f
      val baseDt = (0 until nSamples).map(i => actionScaler.unscale(predScaled.getFloat(i.toLong, 0L))).toArray
      val sensitivities = Array.ofDim[Double](nSamples, stateDim)

      (0 until stateDim).foreach { k =>
        val perturbedPlus = statesArr.map { row =>
          val c = row.clone()
          c(k) = c(k) + eps
          c
        }
        val perturbedMinus = statesArr.map { row =>
          val c = row.clone()
          c(k) = c(k) - eps
          c
        }
        val outPlus = actor.evaluate(sub.create(perturbedPlus))
        val outMinus = actor.evaluate(sub.create(perturbedMinus))
        (0 until nSamples).foreach { i =>
          val dtPlus = actionScaler.unscale(outPlus.getFloat(i.toLong, 0L))
          val dtMinus = actionScaler.unscale(outMinus.getFloat(i.toLong, 0L))
          sensitivities(i)(k) = math.abs(dtPlus - dtMinus) / (2.0 * eps)
        }
      }

      val embFile = new java.io.PrintWriter(new java.io.File(outDir, s"embeddings_weight-$weight.csv"))
      try {
        val featNames = Seq(
          "deltaLatest", "deltaPrev", "selfDeltaMag", "maxNeighDeltaMag", "neighChangedRatio", "bfResidual",
          "childrenRatio", "minNeighAction", "weight"
        )
        val sensNames = featNames.map(n => s"sens_$n")
        val hidNames = (0 until hiddenDim).map(h => s"h_$h")
        embFile.println((featNames ++ Seq("predDt", "qValue", "regime") ++ sensNames ++ hidNames).mkString(","))

        (0 until nSamples).foreach { i =>
          val s = statesArr(i)
          val dt = baseDt(i)
          val qv = qTensor.getFloat(i.toLong, 0L)
          val selfActive = math.abs(s(0)) > 0.01f || math.abs(s(1)) > 0.01f || s(2) > 0.05f
          val neighActive = s(3) > 0.02f || s(4) > 0.01f || s(5) > 0.05f
          val isBackbone = s(6) > 0.15f
          val regime =
            if (selfActive && isBackbone) "Backbone_Wave"
            else if (selfActive) "Leaf_Wave"
            else if (neighActive) "Neighbor_EarlyWarning"
            else "Steady_State"
          val hVec = (0 until hiddenDim).map(h => f"${hiddenTensor.getFloat(i.toLong, h.toLong)}%.5f")
          val sensVec = sensitivities(i).map(v => f"$v%.5f")
          val sVec = s.map(v => f"$v%.5f")
          embFile.println((sVec ++ Seq(f"$dt%.5f", f"$qv%.5f", regime) ++ sensVec ++ hVec).mkString(","))
        }
      } finally
        embFile.close()

      // Also export a 2D Policy Response Surface grid for beta = weight
      val surfFile = new java.io.PrintWriter(new java.io.File(outDir, s"surface_weight-$weight.csv"))
      try {
        surfFile.println("selfDeltaMag,maxNeighDeltaMag,childrenRatio,minNeighAction,predDt,qValue")
        val gridVals = (0 to 25).map(_ / 25.0f)
        for {
          selfMag <- gridVals
          neighMag <- gridVals
          childR <- Seq(0.0f, 0.5f)
        } {
          val deltaFlag = if (selfMag > 0.05f) 1.0f else 0.0f
          val neighRatio = if (neighMag > 0.05f) math.min(1.0f, neighMag * 0.8f) else 0.0f
          val minAct = if (neighMag > 0.1f) 0.3f else 0.8f
          val row =
            Array(deltaFlag, 0.0f, selfMag, neighMag, neighRatio, neighMag * 0.5f, childR, minAct, weight.toFloat)
          val tIn = sub.create(row, new Shape(1, stateDim))
          val scOut = actor.evaluate(tIn)
          val qOut = critic.evaluate(tIn.concat(scOut, 1))
          val dt = actionScaler.unscale(scOut.getFloat(0L))
          val qv = qOut.getFloat(0L)
          surfFile.println(f"$selfMag%.4f,$neighMag%.4f,$childR%.2f,$minAct%.2f,$dt%.5f,$qv%.5f")
        }
      } finally
        surfFile.close()
    }
  }
}

object ActorInspection {

  /** Where the inspections of the experiments are exported. */
  val DefaultDirectory: File = new File("build/exports/embeddings")

  /** Number of transitions of each regime that are inspected. */
  val SamplesPerRegime: Int = 400
}
