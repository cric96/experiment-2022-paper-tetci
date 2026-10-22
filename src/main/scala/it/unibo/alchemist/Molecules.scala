package it.unibo.alchemist

import it.unibo.alchemist.model.Node
import it.unibo.alchemist.model.scafi.nodes.SimpleNodeManager

/** Typed access to the molecules of a node: the cast to the expected type happens here, at the Alchemist boundary. */
object Molecules:
  extension [T](node: Node[T])
    def valueOf[A](molecule: String): Option[A] = SimpleNodeManager[T](node).getOption[A](molecule)
