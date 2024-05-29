package it.unibo.alchemist.loader.`export`

import it.unibo.alchemist.loader.`export`.Extractor
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.alchemist.model.interfaces.{Actionable, Environment, Time}

import java.util
import scala.jdk.CollectionConverters.{CollectionHasAsScala, MapHasAsJava}

class DecayVariableExtractor() extends Extractor[Double] {
  override def getColumnNames: util.List[String] = java.util.List.of("epsilon")
  var memoized: Option[CentralAgent[_, _]] = None
  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      l: Long
  ): util.Map[String, Double] = {
    if (memoized.isEmpty) {
      memoized = environment.getGlobalReactions.asScala
        .find(_.isInstanceOf[CentralAgent[_, _]])
        .map(_.asInstanceOf[CentralAgent[_, _]])
    }

    getColumnNames.asScala.map(variable => variable -> memoized.get.decayVariableOf(variable)).toMap.asJava
  }

}
