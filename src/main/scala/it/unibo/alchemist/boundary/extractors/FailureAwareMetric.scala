package it.unibo.alchemist.boundary.extractors

import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.boundary.Extractor
import it.unibo.alchemist.model.reactions.SingleBlinker
import it.unibo.alchemist.model.{Environment, Reaction, Time}

import java.util
import scala.jdk.CollectionConverters.*

/** Sum/mean of an episode counter, retaining failed nodes' work and the initial population denominator. */
final class FailureAwareMetric(molecule: String, mean: Boolean) extends Extractor[Double]:
  private val column = s"$molecule[${if mean then "mean" else "sum"}]"

  override def getColumnNames: util.List[String] = List(column).asJava

  override def extractData[T](
      environment: Environment[T, ?],
      reaction: Reaction[T],
      time: Time,
      step: Long
  ): util.Map[String, Double] =
    val nodes = environment.getNodes.getCurrent.asScala.toSeq
    val retired = environment.getReactions.getCurrent.asScala.collectFirst { case blinker: SingleBlinker[?, ?] =>
      blinker.failedMetrics
    }
    val current = nodes.flatMap(_.valueOf[Number](molecule)).map(_.doubleValue()).filter(_.isFinite)
    val total = current.sum + retired.fold(0.0)(_.totalOf(molecule))
    val population = current.size + retired.fold(0)(_.failedNodes)
    val value = if mean then if population == 0 then Double.NaN else total / population else total
    Map(column -> value).asJava
