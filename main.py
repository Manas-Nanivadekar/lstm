import torch
import torch.nn as nn
import math


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.b_h = nn.Parameter(torch.randn(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, x_t, h_prev):
        h_t = torch.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
        return h_t
