import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

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

# viz
lstm_cell = LSTM(input_size=5, hidden_size=3)
sequence_length = 10

h_t = torch.zeros(1, 3)
c_t = torch.zeros(1, 3)

# Store gate values over time
forget_gates = []
input_gates = []
output_gates = []
cell_states = []
hidden_states = []

for t in range(sequence_length):
    x_t = torch.randn(1, 5)
    
    # Manual forward to capture gates
    combined = torch.cat([x_t, h_t], dim=1)
    gates = combined @ lstm_cell.W + lstm_cell.b
    f, i, g, o = gates.chunk(4, dim=1)
    
    f = torch.sigmoid(f)
    i = torch.sigmoid(i)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    
    c_t = f * c_t + i * g
    h_t = o * torch.tanh(c_t)
    
    # Store values
    forget_gates.append(f.squeeze().detach().numpy())
    input_gates.append(i.squeeze().detach().numpy())
    output_gates.append(o.squeeze().detach().numpy())
    cell_states.append(c_t.squeeze().detach().numpy())
    hidden_states.append(h_t.squeeze().detach().numpy())

# Convert to arrays
forget_gates = np.array(forget_gates)
input_gates = np.array(input_gates)
output_gates = np.array(output_gates)
cell_states = np.array(cell_states)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(forget_gates)
axes[0].set_title('Forget Gate (f_t) - What to keep from memory')
axes[0].set_ylabel('Gate value')
axes[0].legend([f'Unit {i}' for i in range(3)])
axes[0].grid(True, alpha=0.3)

axes[1].plot(input_gates)
axes[1].set_title('Input Gate (i_t) - How much new info to add')
axes[1].set_ylabel('Gate value')
axes[1].legend([f'Unit {i}' for i in range(3)])
axes[1].grid(True, alpha=0.3)

axes[2].plot(output_gates)
axes[2].set_title('Output Gate (o_t) - What to expose as output')
axes[2].set_ylabel('Gate value')
axes[2].legend([f'Unit {i}' for i in range(3)])
axes[2].grid(True, alpha=0.3)

axes[3].plot(cell_states)
axes[3].set_title('Cell State (c_t) - Long-term memory')
axes[3].set_ylabel('Cell value')
axes[3].set_xlabel('Timestep')
axes[3].legend([f'Unit {i}' for i in range(3)])
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./lstm_gates_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualization saved!")