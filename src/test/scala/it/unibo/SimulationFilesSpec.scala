package it.unibo

import it.unibo.alchemist.loader.LoadAlchemist
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File
import scala.jdk.CollectionConverters._

/** The Scala snippets embedded in the YAML files are compiled only when a simulation starts: load every file (and every
  * algorithm of the generalization scenario) so that a broken constructor or a renamed class fails the build.
  */
class SimulationFilesSpec extends AnyFlatSpec with Matchers {
  private val simulationFiles =
    new File("src/main/yaml").listFiles().filter(_.getName.endsWith(".yml")).sortBy(_.getName)
  private val generalizationAlgorithms = 0 to 9

  private def load(file: File, variables: Map[String, Any] = Map.empty): Int =
    LoadAlchemist.from(file).getWith[Any, Nothing](variables.asJava).getEnvironment.getNodeCount

  simulationFiles.foreach { file =>
    file.getName should "load with its default variables" in {
      load(file) should be > 0
    }
  }

  "simulationGeneralization.yml" should "instantiate every algorithm" in {
    val file = new File("src/main/yaml/simulationGeneralization.yml")
    generalizationAlgorithms.foreach(algorithm => load(file, Map("algoId" -> algorithm)) should be > 0)
  }
}
