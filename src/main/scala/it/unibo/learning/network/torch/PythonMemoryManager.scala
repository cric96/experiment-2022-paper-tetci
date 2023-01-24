package it.unibo.learning.network.torch

import me.shadaj.scalapy.py

object PythonMemoryManager {
  def session(): Session = new Session()
  class Session() {
    var elements: scala.collection.mutable.Buffer[py.Any] = scala.collection.mutable.Buffer.empty
    implicit class RichPy(any: py.Any) {
      def record(): py.Any = {
        elements.addOne(any)
        any
      }
    }
    implicit class RichPyDynamic(any: py.Dynamic) {
      def record(): py.Dynamic = {
        elements.addOne(any)
        any
      }
    }
    def clear(): Unit = {
      elements.foreach(_.del())
      elements.clear()
    }
  }
}
