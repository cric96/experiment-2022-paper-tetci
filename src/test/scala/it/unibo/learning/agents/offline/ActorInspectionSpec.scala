package it.unibo.learning.agents.offline

import ai.djl.nn.Activation
import it.unibo.learning.network.NeuralNetworkRL.SpatialTemporal
import it.unibo.learning.network.{MinMaxActionScaler, TorchNetwork}
import it.unibo.testing.AgentStates
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File
import java.nio.file.Files
import java.security.MessageDigest
import scala.util.Random

class ActorInspectionSpec extends AnyFlatSpec with Matchers {
  private val stateDim = SpatialTemporal.FeatureDim
  private val hiddenLayers = Seq(8, 8)
  private val weight = 0.95

  private def exportedFiles(): Seq[File] = {
    val actor = TorchNetwork.createMLP("actor", stateDim, hiddenLayers, 1, Some(Activation.tanhBlock()), seed = 11)
    val critic = TorchNetwork.createMLP("critic", stateDim + 1, hiddenLayers, 1, seed = 12)
    val random = new Random(13)
    val samples = AgentStates.fixtures.map(SpatialTemporal.encode(_, weight, 5.0)).toVector ++
      Vector.fill(6)(Array.fill(stateDim)(random.nextFloat() * 2 - 1))
    val directory = Files.createTempDirectory("inspection").toFile
    new ActorInspection(actor, critic, MinMaxActionScaler(0.5, 5.0), stateDim, directory)
      .export(samples, weight)
    Seq(s"embeddings_weight-$weight.csv", s"surface_weight-$weight.csv").map(new File(directory, _))
  }

  private def sha256(file: File): String =
    MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file.toPath)).map("%02x".format(_)).mkString

  "The actor inspection" should "keep exporting the same embeddings and policy surface" in {
    exportedFiles().map(sha256) shouldBe Seq(
      "5a2a7940d197700c9e7b7e2008ba41e975212b3122bbe97387ef12f5df0c2244",
      "653709d17143efa4f0d8482b622c6fbac373b6709640eaf704330c4a634dc4b4"
    )
  }
}
