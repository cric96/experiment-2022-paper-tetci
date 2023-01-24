package it.unibo.learning.network.torch

import me.shadaj.scalapy.interpreter.CPythonInterpreter
import me.shadaj.scalapy.py

object util {
  CPythonInterpreter.execManyLines(
    """
      |import torch
      |import gc
      |def get_all_tensor():
      |    tensors = []
      |    for obj in gc.get_objects():
      |      try:
      |        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
      |            tensors.append(obj)
      |      except:
      |        pass
      |    return tensors
      |""".stripMargin
  )

  def getAllTensors(): Seq[py.Dynamic] =
    py.Dynamic.global.get_all_tensor().as[Seq[py.Dynamic]]
}
