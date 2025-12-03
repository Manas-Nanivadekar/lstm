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

# smoke test
input_size = 10
hidden_size = 20
batch_size = 3

rnn = RNN(input_size, hidden_size)

h_0 = torch.zeros(batch_size, hidden_size)

sequence_length = 5
x_sequence = torch.randn(sequence_length, batch_size, input_size)

h_t = h_0
for t in range(sequence_length):
    x_t = x_sequence[t]
    h_t = rnn(x_t, h_t)
    print(f"Timestep {t}: h_t shape = {h_t.shape}")

print(f"\nFinal hidden state: {h_t.shape}")
