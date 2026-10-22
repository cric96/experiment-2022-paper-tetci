package it.unibo.alchemist

/** Adapters between Scala and the Kotlin API of Alchemist. */
object KotlinInterop:

  /** A Kotlin `() -> Unit`, e.g. a command for `Simulation.schedule`. */
  def command(body: => Unit): kotlin.jvm.functions.Function0[kotlin.Unit] = () =>
    body
    kotlin.Unit.INSTANCE
