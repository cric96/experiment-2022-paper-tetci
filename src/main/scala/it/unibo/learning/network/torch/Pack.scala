package it.unibo.learning.network.torch

import me.shadaj.scalapy.py

object Pack {
  import me.shadaj.scalapy.interpreter.CPythonInterpreter
  CPythonInterpreter.execManyLines(
    """import torch
      |from torch import nn
      |class Pack(nn.Module):
      | def __init__(self):
      |   super(Pack, self).__init__()
      | def forward(self,x):
      |   # Output shape (batch, features, hidden)
      |   tensor, memory = x
      |   memory.detach()
      |   del memory
      |   # Reshape shape (batch, hidden)
      |   return tensor[:, -1, :]
      |""".stripMargin
  )
  def apply(): py.Dynamic = py.Dynamic.global.Pack()
}
