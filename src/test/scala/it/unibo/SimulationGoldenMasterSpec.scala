package it.unibo

import it.unibo.testing.{PyTorchNumerics, Simulations}
import it.unibo.testing.Simulations.Fingerprint
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Pins the observable behaviour of every simulation file, so that refactorings can be proven behaviour-preserving.
  *
  * If a change is *meant* to alter the dynamics, update the expected fingerprints and say so in the commit message.
  */
class SimulationGoldenMasterSpec extends AnyFlatSpec with Matchers {

  /** End of the first episode, end of the second one (after one training), shortly after the second training. */
  private val checkpoints = Seq(199.5, 399.5, 450.0)
  private val seededWeight = Map("random" -> 2.0, "weight" -> 0.95)

  private val scenarios = Seq(
    (
      "simulation",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 6730, 339.0, -537.3574207793971, 548.4962547355789, 29134.200526238164),
        Fingerprint(399.5, 100, 6836, 332.0, -458.54755172420505, 552.0817465786448, 27593.35841691538),
        Fingerprint(450.0, 100, 1775, 327.0, -115.52835320140167, 120.43516580125328, 21769.281388504758)
      )
    ),
    (
      "simulationBaseline",
      seededWeight + ("actions" -> 2.0),
      Seq(
        Fingerprint(199.5, 100, 10001, 200.0, -345.73566074382495, 458.58080778236433, 29134.200526238164),
        Fingerprint(399.5, 100, 9950, 200.0, -338.46417032471066, 393.45005323429154, 27593.35841691538),
        Fingerprint(450.0, 100, 2500, 200.0, -90.81108000116934, 148.4721343753803, 21769.281388504758)
      )
    ),
    (
      "simulationDeepOffline",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 6758, 375.0, -539.8665692167757, 497.6619159344896, 29134.200526238164),
        Fingerprint(399.5, 100, 6767, 324.0, -530.037226683669, 571.6383631918037, 27593.35841691538),
        Fingerprint(450.0, 100, 1713, 334.0, -152.13910006928734, 138.96864363126684, 21769.281388504758)
      )
    ),
    (
      "simulationDeepOfflineContinous",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 7292, 331.45582151834907, -507.9901308219393, 424.9470139333739, 29134.200526238164),
        Fingerprint(399.5, 100, 7396, 334.59879806852393, -482.997547026751, 667.7207035385267, 27593.35841691538),
        Fingerprint(450.0, 100, 1870, 293.89974493709315, -128.59133756439937, 94.89714587546814, 21769.281388504758)
      )
    ),
    (
      "simulationDeepOfflineGNN",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 7351, 316.7691666401433, -487.53396008740276, 409.82392553801986, 29134.200526238164),
        Fingerprint(399.5, 100, 7439, 305.62218051764745, -476.5574176106515, 742.8903190724571, 27593.35841691538),
        Fingerprint(450.0, 100, 1926, 315.9998309009481, -143.57793046670193, 130.01861882186898, 21769.281388504758)
      )
    ),
    (
      "simulationGeneralization", // algoId 3 = HeuristicAdaptiveAgent, progId 0 = gradient
      Map("random" -> 2.0, "algoId" -> 3, "progId" -> 0),
      Seq(
        Fingerprint(199.5, 100, 7237, 286.91588785046696, -253.33736269840406, 628.2784813724719, 29134.200526238164),
        Fingerprint(399.5, 100, 7226, 286.91588785046696, -262.11551587425464, 551.8848349293694, 27593.35841691538),
        Fingerprint(450.0, 100, 1796, 286.91588785046696, -74.94979939870588, 212.9950712768299, 21769.281388504758)
      )
    )
  )

  /** The scenarios whose dynamics depend on networks trained with PyTorch. */
  private val neuralScenarios =
    Set("simulationDeepOffline", "simulationDeepOfflineContinous", "simulationDeepOfflineGNN")

  scenarios.foreach { case (yaml, variables, expected) =>
    val behaviour = yaml should "keep its observable dynamics"
    def check(): Unit = Simulations.run(yaml, variables, checkpoints) shouldBe expected
    if neuralScenarios.contains(yaml) then behaviour taggedAs PyTorchNumerics in check() else behaviour in check()
  }

  /** After the training episodes, the generalization scenario switches to its target program and cycles β. */
  private val afterSwitch = Seq(1999.5, 2199.5, 2250.0)
  private val endOfTraining =
    Fingerprint(1999.5, 100, 7200, 286.91588785046696, -259.31281611292957, 717.4457181835604, 23571.475879998467)

  private val unseenPrograms = Seq(
    (
      1,
      "hop gradient",
      Seq(
        endOfTraining,
        Fingerprint(2199.5, 100, 4460, 477.35849056603695, -511.9761246186995, 859.2952594339498, 575.0),
        Fingerprint(2250.0, 100, 1100, 477.35849056603695, -149.19313370121918, 381.886792452824, 401.0)
      )
    ),
    (
      2,
      "broadcast",
      Seq(
        endOfTraining,
        Fingerprint(2199.5, 100, 4311, 477.35849056603695, -468.29652298336913, 3842.735849056545, 1108000.0),
        Fingerprint(2250.0, 100, 1100, 477.35849056603695, -196.25797555443094, 1427.3018867924297, 1267000.0)
      )
    ),
    (
      3,
      "collect",
      Seq(
        endOfTraining,
        Fingerprint(2199.5, 100, 4625, 477.35849056603695, -668.4722625192188, 630.8653478773524, 675.0),
        Fingerprint(2250.0, 100, 1193, 477.35849056603695, -207.60298810259508, 111.58761792452705, 501.0)
      )
    )
  )

  unseenPrograms.foreach { case (program, name, expected) =>
    "simulationGeneralization" should s"keep its dynamics on an unseen program ($name)" in {
      val variables = Map("random" -> 2.0, "algoId" -> 3, "progId" -> program)
      Simulations.run("simulationGeneralization", variables, afterSwitch) shouldBe expected
    }
  }
}
