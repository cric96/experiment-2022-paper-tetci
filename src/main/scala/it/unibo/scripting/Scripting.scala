package it.unibo.scripting

/** `variable.as[Double]`: the variables of the YAML files are untyped, the Scala snippets cast them where needed. */
extension (value: Any) def as[E]: E = value.asInstanceOf[E]
