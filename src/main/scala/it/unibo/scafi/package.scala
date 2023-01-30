package it.unibo

package object scafi {
  implicit class richDouble(value: Double) {
    def convertIfInfinite = if (!value.isFinite) {
      100
    } else {
      value
    }
  }
}
