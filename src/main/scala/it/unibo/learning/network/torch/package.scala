package it.unibo.learning.network

import me.shadaj.scalapy.py

package object torch {
  val torch = py.module("torch")
  val nn = py.module("torch.nn")
  val optim = py.module("torch.optim")
  val log = py.module("torch.utils.tensorboard")
  val writer = log.SummaryWriter()
  val geometric = py.module("torch_geometric")
  val d3rlpy = py.module("d3rlpy")
  val tensordict = py.module("tensordict")
  val numpy = py.module("numpy")
}
