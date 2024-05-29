package it.unibo.alchemist.loader.`export`

import it.unibo.alchemist.loader.`export`.Extractor
import it.unibo.alchemist.model.interfaces.{Actionable, Environment, Time}

import java.util

class EpisodeInfoExtractor extends Extractor[Double] {

  override def getColumnNames: util.List[String] = util.List.of("reward", "error")

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, Double] = ???
}
