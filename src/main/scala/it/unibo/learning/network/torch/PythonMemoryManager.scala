package it.unibo.learning.network.torch

import me.shadaj.scalapy.py

object PythonMemoryManager {
  val pyUnsafe = py.Dynamic.global
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
      val iterator = elements.iterator
      while (iterator.hasNext) {
        try {
          val elem = iterator.next()
          if (pyUnsafe.isinstance(elem, pyUnsafe.dict).as[Boolean]) {
            elem.as[py.Dynamic].clear()
          } else if (pyUnsafe.isinstance(elem, pyUnsafe.list).as[Boolean]) {
            elem.as[py.Dynamic].clear()
          }
          elem.del()
        } catch {
          case _ =>
        }
      }
      elements.clear()
    }
  }
}
