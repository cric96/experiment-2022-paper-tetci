package it.unibo.alchemist.loader.`export`

import it.unibo.alchemist.loader.`export`.DecayVariableExtractor._
import it.unibo.alchemist.model.implementations.timedistributions.reactions.CentralAgent
import it.unibo.alchemist.model.interfaces.{Actionable, Environment, Time}

import java.util
import scala.jdk.CollectionConverters._

/** Exports the current value of the learner's decaying hyper-parameters (0 when the learner has none). */
final class DecayVariableExtractor extends Extractor[Double] {
  override def getColumnNames: util.List[String] = Columns.asJava

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: Time,
      step: Long
  ): util.Map[String, Double] = {
    val centralAgent = environment.getGlobalReactions.iterator().asScala.collectFirst {
      case agent: CentralAgent[_, _] => agent
    }
    Columns.map(name => name -> centralAgent.fold(0.0)(_.decayVariableOf(name))).toMap.asJava
  }
}

object DecayVariableExtractor {
  val Columns: List[String] = List("epsilon")
}
