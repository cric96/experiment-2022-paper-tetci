package it.unibo.learning.network

import me.shadaj.scalapy.py

object RDQN {
  import me.shadaj.scalapy.interpreter.CPythonInterpreter
  CPythonInterpreter.execManyLines(
    """
      |import torch
      |from torch import nn
      |from torch.autograd import Variable
      |class RDQN(nn.Module):
      |    def __init__(self, output_size, input_size, hidden_size, num_layers, seq_length):
      |        super().__init__()
      |        torch.backends.cudnn.enabled=False
      |        torch.backends.cudnn.benchmark=False
      |        self.output_size = output_size #number of classes
      |        self.num_layers = num_layers #number of layers
      |        self.input_size = input_size #input size
      |        self.hidden_size = hidden_size #hidden state
      |        self.seq_length = seq_length #sequence length
      |
      |        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
      |                          num_layers=num_layers) #lstm
      |        self.fc_1 = nn.Linear(hidden_size, hidden_size) #fully connected 1
      |        self.fc = nn.Linear(hidden_size, output_size) #fully connected last layer
      |        self.relu = nn.ReLU()
      |    
      |    def forward(self,x):
      |        h_0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device) #hidden state
      |        # Propagate input through LSTM
      |        output, hn = self.gru(x, h_0) #lstm with input, hidden, and internal state
      |        output = output[:, -1, :]
      |        out = self.relu(output)
      |        out = self.fc_1(out) #first Dense
      |        out = self.relu(out) #relu
      |        out = self.fc(out) #Final Output
      |        return out
      | """.stripMargin
  )
  def apply(input: Int, hidden: Int, output: Int, sequenceSize: Int): py.Dynamic =
    py.Dynamic.global.RDQN(output, input, hidden, 1, sequenceSize)
}
