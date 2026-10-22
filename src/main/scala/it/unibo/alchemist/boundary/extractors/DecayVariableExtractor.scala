package it.unibo.alchemist.boundary.extractors

import it.unibo.alchemist.boundary.Extractor
import it.unibo.alchemist.boundary.extractors.DecayVariableExtractor.*
import it.unibo.alchemist.model.reactions.CentralAgent
import it.unibo.alchemist.model.{Environment, Reaction, Time}

import java.util
import scala.jdk.CollectionConverters.*

/** Exports the current value of the learner's decaying hyper-parameters (0 when the learner has none). */
final class DecayVariableExtractor extends Extractor[Double]:
  override def getColumnNames: util.List[String] = Columns.asJava

  override def extractData[T](
      environment: Environment[T, ?],
      reaction: Reaction[T],
      time: Time,
      step: Long
  ): util.Map[String, Double] =
    val centralAgent = environment.getReactions.getCurrent.asScala.collectFirst { case agent: CentralAgent[?, ?] =>
      agent
    }
    Columns.map(name => name -> centralAgent.fold(0.0)(_.decayVariableOf(name))).toMap.asJava

object DecayVariableExtractor:
  val Columns: List[String] = List("epsilon")
