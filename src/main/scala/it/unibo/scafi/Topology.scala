package it.unibo.scafi

/** The communication graph of the network: the node ids, in deployment order, and the neighbours of each node with the
  * length of the link.
  */
final case class Topology(nodes: Seq[Int], neighbours: Map[Int, Seq[(Int, Double)]]) {
  def neighboursOf(node: Int): Seq[(Int, Double)] = neighbours.getOrElse(node, Seq.empty)
}
