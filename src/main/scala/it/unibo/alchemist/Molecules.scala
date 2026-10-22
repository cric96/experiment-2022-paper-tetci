package it.unibo.alchemist

import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.interfaces.Node

/** Typed access to the molecules of a node: the cast to the expected type happens here, at the Alchemist boundary. */
object Molecules {
  implicit final class MoleculeAccess[T](private val node: Node[T]) extends AnyVal {
    def valueOf[A](molecule: String): Option[A] = {
      val manager = new SimpleNodeManager[T](node)
      Option.when(manager.has(molecule))(manager.get[A](molecule))
    }
  }
}
