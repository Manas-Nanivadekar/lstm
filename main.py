import torch
import torch.nn as nn
import math


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(torch.randn(input_size + hidden_size, 4* hidden_size))
        self.b = nn.Parameter(torch.randn(4 * hidden_size))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x_t, h_prev, c_prev):
        batch_size = x_t.size(0)

        # Concatenate input and hidden state
        combined = torch.cat([x_t, h_prev], dim=1)

        # One matrix multiplication for all gates
        gates = combined @ self.W + self.b

        f_t, i_t, g_t, o_t = gates.chunk(4, dim=1)

        f_t = torch.sigmoid(f_t) # Forget Gate
        i_t = torch.sigmoid(i_t) # Input Gate
        g_t = torch.tanh(g_t) # Candidate cell state
        o_t = torch.sigmoid(o_t) # Output gate

        # Update cell state
        c_t = f_t * c_prev + i_t * g_t

        # Update hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

# smoke test
input_size = 10
hidden_size = 20
batch_size = 3

lstm_cell = LSTM(input_size, hidden_size)

h_0 = torch.zeros(batch_size, hidden_size)
c_0 = torch.zeros(batch_size, hidden_size)

sequence_length = 5
x_sequence = torch.randn(sequence_length, batch_size, input_size)

h_t, c_t = h_0, c_0
for t in range(sequence_length):
    x_t = x_sequence[t]
    h_t, c_t = lstm_cell(x_t, h_t, c_t)
    print(f"Timestep {t}: h_t shape = {h_t.shape}, c_t shape = {c_t.shape}")

print(f"\nFinal hidden state: {h_t.shape}")
print(f"Final cell state: {c_t.shape}")