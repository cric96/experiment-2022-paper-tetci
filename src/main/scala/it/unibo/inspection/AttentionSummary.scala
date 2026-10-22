package it.unibo.inspection

/** Where the attention of a graph-attention policy goes, for a node at `origin`.
  *
  * @param maxAttention
  *   the largest attention weight
  * @param entropy
  *   the entropy of the attention weights (low: focused on few neighbours)
  * @param direction
  *   the attention-weighted sum of the vectors from the node to its neighbours
  * @param focus
  *   the position of the most attended neighbour (the node itself if it has none)
  */
final case class AttentionSummary(
    maxAttention: Float,
    entropy: Double,
    direction: (Double, Double),
    focus: (Double, Double)
) {
  private def attending(origin: (Double, Double), weight: Float, position: (Double, Double)): AttentionSummary = {
    val (x, y) = position
    val pointed = copy(direction = (direction._1 + weight * (x - origin._1), direction._2 + weight * (y - origin._2)))
    if (weight > maxAttention) pointed.copy(maxAttention = weight, focus = position) else pointed
  }
}

object AttentionSummary {

  /** Attention weights below this value do not contribute to the entropy. */
  val NegligibleAttention: Float = 1e-6f

  /** Summarizes the `attention` paid to neighbours whose positions are `positions` (unknown ones count only in the
    * entropy).
    */
  def of(origin: (Double, Double), attention: Seq[Float], positions: Seq[Option[(Double, Double)]]): AttentionSummary =
    attention.zip(positions).foldLeft(AttentionSummary(0.0f, 0.0, (0.0, 0.0), origin)) {
      case (summary, (weight, position)) =>
        val withEntropy =
          if (weight > NegligibleAttention) summary.copy(entropy = summary.entropy - weight * math.log(weight))
          else summary
        position.fold(withEntropy)(withEntropy.attending(origin, weight, _))
    }
}
