package it.unibo

/** Helpers for the Scala snippets embedded in the YAML files. */
package object scripting {

  /** `variable.as[Double]`: the YAML variables are untyped, the snippets cast them to the expected type. */
  implicit final class Unsafe(private val value: Any) extends AnyVal {
    def as[E]: E = value.asInstanceOf[E]
  }
}
