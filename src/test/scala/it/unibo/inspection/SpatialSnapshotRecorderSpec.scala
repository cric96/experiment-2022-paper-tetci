package it.unibo.inspection

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File
import java.nio.file.Files
import scala.jdk.CollectionConverters.*

class SpatialSnapshotRecorderSpec extends AnyFlatSpec with Matchers:
  private def linesOf(directory: File, name: String): Seq[String] =
    Files.readAllLines(new File(directory, name).toPath).asScala.toSeq

  "The spatial snapshots" should "not be written before something is recorded" in {
    val directory = Files.createTempDirectory("spatial").toFile
    SpatialSnapshotRecorder(directory, weight = 0.8).flush()
    directory.list() shouldBe empty
  }

  they should "contain one row per node and time, and the links of the network once" in {
    val directory = Files.createTempDirectory("spatial").toFile
    val recorder = SpatialSnapshotRecorder(directory, weight = 0.8)
    val node = NodeView(3, 1.25, -5.5, isSource = true, 12.25, Double.PositiveInfinity, period = 1.5, ticks = 7)
    val links = Seq(Link(1, 2, (0.0, 0.0), (10.5, 0.0)), Link(2, 1, (10.5, 0.0), (0.0, 0.0)))
    recorder.record(2.01, Seq(node), links)
    recorder.record(3.01, Seq(node.copy(ticks = 8)), Seq.empty)
    recorder.flush()
    linesOf(directory, "spatial_snapshots_weight-0.8.csv") shouldBe Seq(
      "tEp,nodeId,x,y,isSource,gradient,groundTruth,nextWakeUp,error,ticks",
      "2.01,3,1.25,-5.50,1,12.25,600.00,1.5000,587.75,7",
      "3.01,3,1.25,-5.50,1,12.25,600.00,1.5000,587.75,8"
    )
    linesOf(directory, "spatial_edges_weight-0.8.csv") shouldBe Seq("u,v,xU,yU,xV,yV", "1,2,0.00,0.00,10.50,0.00")
  }
end SpatialSnapshotRecorderSpec
