package it.unibo

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.{Environment, Position}

import scala.jdk.CollectionConverters.CollectionHasAsScala

package object alchemist {
  implicit class EnvironmentOps[T](environment: Environment[T, _]) {
    def getClonedOfThis(other: Int): SimpleNodeManager[T] = {
      val node = environment.getNodeByID(other)
      val position = environment.getPosition(node)
      new SimpleNodeManager(
        environment.getNodes.asScala
          .find(n => environment.getPosition(n) == environment.getPosition(node) && n.getId != node.getId)
          .get
      )
    }
  }
}
