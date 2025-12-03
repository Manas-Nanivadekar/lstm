import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(torch.randn(input_size + hidden_size, 4 * hidden_size))
        self.b = nn.Parameter(torch.randn(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, x_t, h_prev, c_prev):
        batch_size = x_t.size(0)

        # Concatenate input and hidden state
        combined = torch.cat([x_t, h_prev], dim=1)

        # One matrix multiplication for all gates
        gates = combined @ self.W + self.b

        f_t, i_t, g_t, o_t = gates.chunk(4, dim=1)

        f_t = torch.sigmoid(f_t)  # Forget Gate
        i_t = torch.sigmoid(i_t)  # Input Gate
        g_t = torch.tanh(g_t)  # Candidate cell state
        o_t = torch.sigmoid(o_t)  # Output gate

        # Update cell state
        c_t = f_t * c_prev + i_t * g_t

        # Update hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList(
            [
                LSTMCell(hidden_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, hidden=None):
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            h = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
            c = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_0, c_0 = hidden
            h = [h_0[i] for i in range(self.num_layers)]
            c = [c_0[i] for i in range(self.num_layers)]

        outputs = []

        for t in range(seq_len):
            x_t = x[t]

            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = h[layer]

            outputs.append(h[-1])

        output = torch.stack(outputs, dim=0)

        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        return output, (h_n, c_n)
