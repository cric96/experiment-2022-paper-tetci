package it.unibo.inspection

import it.unibo.util.Csv
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File
import java.nio.file.Files
import scala.jdk.CollectionConverters._

class AttentionSnapshotRecorderSpec extends AnyFlatSpec with Matchers {
  "The attention snapshots" should "contain the node, its attention summary and the first embedding units" in {
    val directory = Files.createTempDirectory("attention").toFile
    val recorder = new AttentionSnapshotRecorder(directory)
    val node = NodeView(3, 1.25, -5.5, isSource = false, 12.25, 12.5, period = 1.5, ticks = 7)
    val summary = AttentionSummary(0.75f, 0.5, (7.5, 1.0), (10.0, 0.0))
    recorder.record(85.01, node, summary, Array.tabulate(20)(_ * 0.5f))
    recorder.flush()
    val lines = Files.readAllLines(new File(directory, "gnn_spatiotemporal_embeddings.csv").toPath).asScala
    lines.head shouldBe "tEp,nodeId,x,y,isSource,gradient,groundTruth,nextWakeUp,error,maxAlpha,attEntropy,attDx,attDy," +
      "bestNbrX,bestNbrY," + (0 until 16).map(unit => s"h_$unit").mkString(",")
    lines(1) shouldBe "85.0,3,1.25,-5.50,0,12.25,12.50,1.5000,0.25,0.7500,0.5000,7.50,1.00,10.00,0.00," +
      (0 until 16).map(unit => Csv.number(unit * 0.5, 4)).mkString(",")
  }
}
