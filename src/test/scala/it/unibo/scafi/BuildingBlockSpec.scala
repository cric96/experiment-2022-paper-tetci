package it.unibo.scafi

import it.unibo.scafi.BuildingBlock._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BuildingBlockSpec extends AnyFlatSpec with Matchers {
  "The program molecule" should "select a building block by code or by name" in {
    Seq(0, 1, 2, 3)
      .map(code => BuildingBlock.fromMolecule(code)) shouldBe Seq(Gradient, HopGradient, Broadcast, Collect)
    BuildingBlock.fromMolecule(2.0) shouldBe Broadcast
    BuildingBlock.fromMolecule("collect") shouldBe Collect
  }

  it should "default to the gradient" in {
    BuildingBlock.fromMolecule(7) shouldBe Gradient
    BuildingBlock.fromMolecule("unknown") shouldBe Gradient
    BuildingBlock.fromMolecule(true) shouldBe Gradient
  }

  "Each building block" should "normalize its error by the scale of its outputs" in {
    BuildingBlock.all.map(_.errorScale) shouldBe Seq(500.0, 10.0, 500.0, 40.0)
  }

  "The broadcast" should "spread a payload derived from the id of the source" in {
    Broadcast.payloadOf(sourceId = 4) shouldBe 50.0
  }

  "The error of a node" should "be the distance from the ground truth" in {
    Gradient.absoluteError(output = 12.0, groundTruth = 10.5) shouldBe 1.5
    Gradient.normalizedError(output = 12.0, groundTruth = 10.5) shouldBe 1.5 / Gradient.errorScale
  }

  it should "be zero when both values are unbounded, and the full scale when only one is" in {
    Collect.absoluteError(Double.PositiveInfinity, Double.PositiveInfinity) shouldBe 0.0
    Collect.absoluteError(3.0, Double.PositiveInfinity) shouldBe Collect.errorScale
    Collect.absoluteError(Double.PositiveInfinity, 3.0) shouldBe Collect.errorScale
  }

  it should "be normalized to at most one" in {
    HopGradient.normalizedError(output = 100.0, groundTruth = 0.0) shouldBe 1.0
  }
}
