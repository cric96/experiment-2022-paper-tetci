package it.unibo.learning.network

import it.unibo.learning.network.torch.geometric
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.Any.from

object GNNDQN {
  def apply(input: Int, hidden: Int, output: Int): py.Dynamic = {
    val module = geometric.nn.Sequential(
      "x, edge_index",
      Seq(
        (geometric.nn.GCN(input, hidden, 1), "x, edge_index -> x".as[py.Any]).as[py.Any],
        torch.nn.ReLU(inplace = true),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(inplace = true),
        torch.nn.Linear(hidden, output)
      ).toPythonCopy
    )
    module
  }
}
