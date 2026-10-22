package it.unibo

import it.unibo.testing.Simulations
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
        Fingerprint(199.5, 100, 6732, 383.0, -606.5045153728707, 2937.8129150948394, 29134.200526238164),
        Fingerprint(399.5, 100, 6871, 348.0, -608.7215661517886, 2109.158930520948, 27593.35841691538),
        Fingerprint(450.0, 100, 1696, 346.0, -140.12271655703935, 1181.5967633131188, 21769.281388504758)
      )
    ),
    (
      "simulationBaseline",
      seededWeight + ("actions" -> 2.0),
      Seq(
        Fingerprint(199.5, 100, 10001, 200.0, -369.16996434993706, 1897.2229074453408, 29134.200526238164),
        Fingerprint(399.5, 100, 9950, 200.0, -378.7466896461848, 1654.332298605182, 27593.35841691538),
        Fingerprint(450.0, 100, 2500, 200.0, -102.36059895780787, 892.4721343753803, 21769.281388504758)
      )
    ),
    (
      "simulationDeepOffline",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 6744, 381.0, -616.4461208778843, 3006.529381407801, 29134.200526238164),
        Fingerprint(399.5, 100, 6862, 339.0, -686.5352128175828, 2259.97930208459, 27593.35841691538),
        Fingerprint(450.0, 100, 1705, 347.0, -166.71969016995124, 1119.5638861017949, 21769.281388504758)
      )
    ),
    (
      "simulationDeepOfflineContinous",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 7349, 320.7762647967006, -587.460878028568, 2320.9240662274624, 29134.200526238164),
        Fingerprint(399.5, 100, 7395, 307.2700774631161, -600.7130000995472, 2228.945168978838, 27593.35841691538),
        Fingerprint(450.0, 100, 1869, 328.8663117600939, -156.560271321831, 786.2982350707624, 21769.281388504758)
      )
    ),
    (
      "simulationDeepOfflineGNN",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 7330, 310.7835610692959, -561.286135769163, 2334.1506317018166, 29134.200526238164),
        Fingerprint(399.5, 100, 7455, 303.6386748309103, -607.8638989389176, 1963.881697256357, 27593.35841691538),
        Fingerprint(450.0, 100, 1912, 325.6109137742949, -170.7844089209014, 1086.779845676876, 21769.281388504758)
      )
    ),
    (
      "simulationGeneralization", // algoId 3 = HeuristicAdaptiveAgent, progId 0 = gradient
      Map("random" -> 2.0, "algoId" -> 3, "progId" -> 0),
      Seq(
        Fingerprint(199.5, 100, 7201, 286.91588785046696, -294.64304813164256, 2015.0455725243496, 29134.200526238164),
        Fingerprint(399.5, 100, 7270, 286.91588785046696, -308.51380045798044, 1795.81110406843, 27593.35841691538),
        Fingerprint(450.0, 100, 1804, 286.91588785046696, -92.67175229558936, 878.540323356459, 21769.281388504758)
      )
    )
  )

  scenarios.foreach { case (yaml, variables, expected) =>
    yaml should "keep its observable dynamics" in {
      Simulations.run(yaml, variables, checkpoints) shouldBe expected
    }
  }
}
