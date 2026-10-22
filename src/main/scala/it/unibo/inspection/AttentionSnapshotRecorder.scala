package it.unibo.inspection

import it.unibo.inspection.AttentionSnapshotRecorder._
import it.unibo.util.Csv

import java.io.File
import scala.collection.mutable.ArrayBuffer

/** Records, for every node over time, where a graph-attention policy looks and its readout embedding. */
final class AttentionSnapshotRecorder(directory: File) {
  private val rows: ArrayBuffer[Seq[String]] = ArrayBuffer.empty

  def record(timeInEpisode: Double, node: NodeView, summary: AttentionSummary, embedding: Array[Float]): Unit =
    rows += Seq(Csv.number(timeInEpisode, TimeDecimals)) ++ node.columns ++ summaryColumns(summary) ++
      embedding.take(EmbeddingUnits).map(Csv.number(_, EmbeddingDecimals))

  /** Writes the file, if something has been recorded. */
  def flush(): Unit = if (rows.nonEmpty) Csv.write(directory, FileName, Header, rows)

  private def summaryColumns(summary: AttentionSummary): Seq[String] =
    Seq(
      Csv.number(summary.maxAttention, AttentionDecimals),
      Csv.number(summary.entropy, AttentionDecimals)
    ) ++ Seq(summary.direction._1, summary.direction._2, summary.focus._1, summary.focus._2)
      .map(Csv.number(_, NodeView.CoordinateDecimals))
}

object AttentionSnapshotRecorder {
  val FileName: String = "gnn_spatiotemporal_embeddings.csv"

  /** Number of embedding units exported. */
  val EmbeddingUnits: Int = 16

  val TimeDecimals: Int = 1
  val AttentionDecimals: Int = 4
  val EmbeddingDecimals: Int = 4

  val Header: Seq[String] = Seq("tEp") ++ NodeView.Header ++
    Seq("maxAlpha", "attEntropy", "attDx", "attDy", "bestNbrX", "bestNbrY") ++ (0 until EmbeddingUnits).map(unit =>
      s"h_$unit"
    )
}
