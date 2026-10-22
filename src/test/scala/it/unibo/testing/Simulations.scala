package it.unibo.testing

import it.unibo.alchemist.boundary.interfaces.OutputMonitor
import it.unibo.alchemist.core.implementations.Engine
import it.unibo.alchemist.loader.LoadAlchemist
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.{Actionable, Environment, Position, Time}
import it.unibo.scafi.Sensors

import java.io.File
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters._

/** Runs the YAML simulations headless and without exporters, to pin their observable behaviour. */
object Simulations {

  /** Aggregated, order-independent view of the state of all the nodes at a given time. */
  final case class Fingerprint(
      time: Double,
      nodes: Int,
      totalTicks: Long,
      wakeUpSum: Double,
      rewardSum: Double,
      errorSum: Double,
      outputSum: Double
  )

  /** Runs `src/main/yaml/<yaml>.yml` and returns a fingerprint of the nodes as soon as each checkpoint is passed. */
  def run(yaml: String, variables: Map[String, Any], checkpoints: Seq[Double]): Seq[Fingerprint] = {
    val loader = LoadAlchemist.from(new File(s"src/main/yaml/$yaml.yml"))
    simulate[Nothing](loader.getWith[Any, Nothing](variables.asJava).getEnvironment, checkpoints.sorted)
  }

  private def simulate[P <: Position[P]](
      environment: Environment[Any, P],
      checkpoints: Seq[Double]
  ): Seq[Fingerprint] = {
    val engine = new Engine[Any, P](environment, new DoubleTime(checkpoints.max))
    val fingerprints = ListBuffer.empty[Fingerprint]
    engine.addOutputMonitor(new OutputMonitor[Any, P] {
      override def stepDone(environment: Environment[Any, P], reaction: Actionable[Any], time: Time, step: Long): Unit =
        checkpoints.drop(fingerprints.size).headOption.filter(_ <= time.toDouble).foreach { checkpoint =>
          fingerprints += fingerprintOf(environment, checkpoint)
        }
    })
    engine.play()
    engine.run()
    engine.getError.ifPresent(error => throw error)
    fingerprints.toList
  }

  private def fingerprintOf(environment: Environment[Any, _], time: Double): Fingerprint = {
    val managers = environment.getNodes.iterator().asScala.toList.map(new SimpleNodeManager[Any](_))
    def sumOf(molecule: String): Double =
      managers.filter(_.has(molecule)).map(_.get[Double](molecule)).filter(_.isFinite).sum
    Fingerprint(
      time = time,
      nodes = managers.size,
      totalTicks = managers.filter(_.has(Sensors.ticks)).map(_.get[Number](Sensors.ticks).longValue()).sum,
      wakeUpSum = sumOf(Sensors.nextWakeUp),
      rewardSum = sumOf(Sensors.accumulatedReward),
      errorSum = sumOf(Sensors.accumulatedError),
      outputSum = sumOf(Sensors.localComputation)
    )
  }
}
