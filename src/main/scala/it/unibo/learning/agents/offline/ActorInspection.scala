package it.unibo.learning.agents.offline

import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.{NDArray, NDManager}
import ai.djl.nn.Block
import it.unibo.learning.TradeOff
import it.unibo.learning.agents.offline.ActorInspection.*
import it.unibo.learning.network.{MinMaxActionScaler, TorchNetwork}
import it.unibo.util.Csv

import java.io.File

/** Post-training inspection of a continuous actor-critic policy, exported for the explainability figures:
  *   - `embeddings_weight-β.csv`: for each inspected state, its features, the predicted period and value, its
  *     [[Regime]], the sensitivity of the period to each feature and the actor's last hidden layer;
  *   - `surface_weight-β.csv`: the predicted period and value on the [[PolicySurface]].
  */
final class ActorInspection(
    actor: TorchNetwork[Block],
    critic: TorchNetwork[Block],
    actionScaler: MinMaxActionScaler,
    stateDim: Int,
    outputDirectory: File
):
  private val rootManager: NDManager = NDManager.newBaseManager()

  def save(states: Vector[Array[Float]], weight: Double): Unit =
    if states.nonEmpty then
      TorchNetwork.withSubManager(rootManager) { manager =>
        exportEmbeddings(states, weight, manager)
        exportSurface(weight, manager)
      }

  private def exportEmbeddings(states: Vector[Array[Float]], weight: Double, manager: NDManager): Unit =
    val input = manager.create(states.toArray)
    val hidden = actor.extractHiddenEmbedding(input)
    val scaledPeriods = actor.evaluate(input)
    val values = critic.evaluate(input.concat(scaledPeriods, 1))
    val sensitivity = sensitivities(states, manager)
    val hiddenSize = hidden.getShape.get(1).toInt
    val rows = states.indices.map { sample =>
      val hiddenUnits = (0 until hiddenSize).map(unit => hidden.getFloat(sample.toLong, unit.toLong).toDouble)
      states(sample).map(Csv.number(_, ValueDecimals)).toSeq ++
        Seq(
          Csv.number(periodAt(scaledPeriods, sample), ValueDecimals),
          Csv.number(values.getFloat(sample.toLong, 0L), ValueDecimals),
          Regime.of(states(sample)).label
        ) ++
        (sensitivity(sample) ++ hiddenUnits).map(Csv.number(_, ValueDecimals))
    }
    Csv.write(outputDirectory, s"embeddings_weight-$weight.csv", embeddingsHeader(hiddenSize), rows)

  /** |∂Δt/∂x_k| estimated with central finite differences, for each state (outer) and feature k (inner). */
  private def sensitivities(states: Vector[Array[Float]], manager: NDManager): IndexedSeq[IndexedSeq[Double]] =
    val byFeature = (0 until stateDim).map { feature =>
      val increased = periodsOf(perturbed(states, feature, SensitivityStep), manager)
      val decreased = periodsOf(perturbed(states, feature, -SensitivityStep), manager)
      increased.zip(decreased).map { case (up, down) => math.abs(up - down) / (2.0 * SensitivityStep) }
    }
    states.indices.map(sample => byFeature.map(_(sample)))

  private def perturbed(states: Vector[Array[Float]], feature: Int, delta: Float): Array[Array[Float]] =
    states.map { state =>
      val copy = state.clone()
      copy(feature) = copy(feature) + delta
      copy
    }.toArray

  private def periodsOf(states: Array[Array[Float]], manager: NDManager): IndexedSeq[Double] =
    val scaledPeriods = actor.evaluate(manager.create(states))
    states.indices.map(periodAt(scaledPeriods, _))

  private def periodAt(scaledPeriods: NDArray, sample: Int): Double =
    actionScaler.unscale(scaledPeriods.getFloat(sample.toLong, 0L).toDouble)

  private def exportSurface(weight: Double, manager: NDManager): Unit =
    val rows = PolicySurface.points.map { point =>
      val input = manager.create(point.features(TradeOff.betaFeature(weight)), new Shape(1, stateDim.toLong))
      val scaledPeriod = actor.evaluate(input)
      val value = critic.evaluate(input.concat(scaledPeriod, 1))
      Seq(
        Csv.number(point.selfChange, GridDecimals),
        Csv.number(point.neighbourChange, GridDecimals),
        Csv.number(point.childrenRatio, RatioDecimals),
        Csv.number(point.shortestNeighbourPeriod, RatioDecimals),
        Csv.number(periodAt(scaledPeriod, 0), ValueDecimals),
        Csv.number(value.getFloat(0L), ValueDecimals)
      )
    }
    Csv.write(outputDirectory, s"surface_weight-$weight.csv", SurfaceHeader, rows)
end ActorInspection

object ActorInspection:

  /** Where the inspections of the experiments are exported. */
  val DefaultDirectory: File = new File("build/exports/embeddings")

  /** Number of transitions of each regime that are inspected. */
  val SamplesPerRegime: Int = 400

  /** Perturbation of a feature used to estimate the sensitivity of the period to it. */
  val SensitivityStep: Float = 0.1f

  /** Decimal digits of the exported values, grid coordinates and ratios. */
  val ValueDecimals: Int = 5
  val GridDecimals: Int = 4
  val RatioDecimals: Int = 2

  /** Column names of the features, as read by the plotting scripts. */
  val FeatureColumns: Seq[String] = Seq(
    "deltaLatest", "deltaPrev", "selfDeltaMag", "maxNeighDeltaMag", "neighChangedRatio", "bfResidual", "childrenRatio",
    "minNeighAction", "weight"
  )

  val SurfaceHeader: Seq[String] =
    Seq("selfDeltaMag", "maxNeighDeltaMag", "childrenRatio", "minNeighAction", "predDt", "qValue")

  private def embeddingsHeader(hiddenSize: Int): Seq[String] =
    FeatureColumns ++ Seq("predDt", "qValue", "regime") ++ FeatureColumns.map(name => s"sens_$name") ++
      (0 until hiddenSize).map(unit => s"h_$unit")
end ActorInspection
