package it.unibo.alchemist.loader.`export`

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.{Actionable, Environment, Time}
import it.unibo.scafi.Sensors

import java.util
import scala.jdk.CollectionConverters.{IteratorHasAsScala, MapHasAsJava}

class EpisodeInfoExtractor extends Extractor[Double] {

  override def getColumnNames: util.List[String] = util.List.of("reward", "unstableTime")

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, Double] = {
    val managers = environment.getNodes.iterator().asScala.map(new SimpleNodeManager[T](_)).toList
    if (managers.isEmpty) {
      Map("reward" -> 0.0, "unstableTime" -> 0.0).asJava
    } else {
      val avgReward = managers
        .map(m => if (m.has(Sensors.accumulatedReward)) m.get[Double](Sensors.accumulatedReward) else 0.0)
        .sum / managers.size
      val avgUnstable = managers
        .map(m => if (m.has("accumulatedUnstableTime")) m.get[Double]("accumulatedUnstableTime") else 0.0)
        .sum / managers.size
      Map("reward" -> avgReward, "unstableTime" -> avgUnstable).asJava
    }
  }
}
