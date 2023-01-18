package it.unibo

import scala.language.implicitConversions

package object scripting {
  implicit class Unsafe(elem: Any) {
    def as[E] = elem.asInstanceOf[E]
  }
}
