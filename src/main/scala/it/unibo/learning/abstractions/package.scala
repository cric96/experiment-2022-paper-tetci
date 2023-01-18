package it.unibo.learning

package object abstractions {
  type Contextual = Any // Any additional information produce by the neural network/policy
  object Contextual {
    def empty: Contextual = () // empty value
  }
}
