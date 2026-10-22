package it.unibo.scafi

/** The self-stabilizing building blocks whose rounds are scheduled by the learned policies.
  *
  * @param code
  *   the value of the `program` molecule that selects the block
  * @param errorScale
  *   the magnitude of the block's outputs, used to normalize the error against the ground truth
  */
sealed abstract class BuildingBlock(val name: String, val code: Int, val errorScale: Double) {

  /** How far `output` is from `groundTruth`: zero if both are unbounded, the whole `errorScale` if only one is. */
  def absoluteError(output: Double, groundTruth: Double): Double =
    if output.isFinite && groundTruth.isFinite then math.abs(output - groundTruth)
    else if !output.isFinite && !groundTruth.isFinite then 0.0
    else errorScale

  /** The absolute error relative to the scale of the block's outputs, saturated at one. */
  def normalizedError(output: Double, groundTruth: Double): Double =
    math.min(1.0, absoluteError(output, groundTruth) / errorScale)
}

object BuildingBlock {

  /** Distance from the nearest source. */
  case object Gradient extends BuildingBlock("gradient", code = 0, errorScale = 500.0)

  /** Number of hops from the nearest source. */
  case object HopGradient extends BuildingBlock("hopGradient", code = 1, errorScale = 10.0)

  /** The source's payload, spread along the gradient. */
  case object Broadcast extends BuildingBlock("broadcast", code = 2, errorScale = 500.0) {

    /** Payload units per source id, so that every source spreads a distinct value. */
    val PayloadPerId: Double = 10.0

    def payloadOf(sourceId: Int): Double = (sourceId + 1).toDouble * PayloadPerId
  }

  /** Number of nodes in the subtree of the gradient's spanning tree rooted in each node. */
  case object Collect extends BuildingBlock("collect", code = 3, errorScale = 40.0)

  val all: Seq[BuildingBlock] = Seq(Gradient, HopGradient, Broadcast, Collect)

  /** The block selected by the value of the `program` molecule, a code or a name; the gradient otherwise. */
  def fromMolecule(value: Any): BuildingBlock = value match {
    case code: Number => all.find(_.code == code.intValue()).getOrElse(Gradient)
    case name: String => all.find(_.name == name).getOrElse(Gradient)
    case _ => Gradient
  }
}
