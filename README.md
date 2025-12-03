# LSTM: Long Short-Term Memory Networks

## Table of Contents
1. [Introduction](#introduction)
2. [The Problem: Vanishing/Exploding Gradients](#the-problem-vanishingexploding-gradients)
3. [Vanilla RNN: Understanding the Basics](#vanilla-rnn-understanding-the-basics)
4. [Why Vanilla RNNs Fail](#why-vanilla-rnns-fail)
5. [LSTM: The Solution](#lstm-the-solution)
6. [LSTM Architecture Deep Dive](#lstm-architecture-deep-dive)
7. [Implementation Details](#implementation-details)
8. [Complete Code](#complete-code)
9. [Key Takeaways](#key-takeaways)

---

## Introduction

Long Short-Term Memory (LSTM) networks are a special kind of Recurrent Neural Network (RNN) designed to learn long-term dependencies. Unlike vanilla RNNs, LSTMs can effectively learn relationships between events separated by hundreds of timesteps.

**Core Question**: Why do we need LSTMs when we already have RNNs?
**Answer**: Vanilla RNNs suffer from the vanishing/exploding gradient problem, making them unable to learn long-term dependencies.

---

## The Problem: Vanishing/Exploding Gradients

### The Challenge
When training neural networks on sequential data, we need gradients to flow backward through time to update weights. In long sequences, this becomes problematic.

### Mathematical Intuition

Consider gradient flow through T timesteps:

```
∂L/∂h₀ = (∂L/∂hₜ) · (∂hₜ/∂hₜ₋₁) · (∂hₜ₋₁/∂hₜ₋₂) · ... · (∂h₁/∂h₀)
```

Each term in this chain involves matrix multiplications. When multiplied T times:
- If eigenvalues > 1 → **Gradients explode** (→ ∞)
- If eigenvalues < 1 → **Gradients vanish** (→ 0)

---

## Vanilla RNN: Understanding the Basics

### The Core Idea
At each timestep, combine current input with previous memory to create new memory.

### The Equation
```
hₜ = tanh(Wₓₕ @ xₜ + Wₕₕ @ hₜ₋₁ + bₕ)
```

Where:
- `xₜ` : Current input at time t
- `hₜ₋₁` : Previous hidden state (memory from time t-1)
- `hₜ` : New hidden state
- `Wₓₕ` : Weight matrix transforming input
- `Wₕₕ` : Weight matrix transforming previous hidden state
- `tanh` : Activation function (squashes values to [-1, 1])

### Why This Design?

**Q: Why not just multiply xₜ and hₜ₋₁ directly?**

A: Dimensionality mismatch! We need weight matrices to:
1. Project both inputs into the same hidden space
2. Allow the network to learn what combinations matter

**Q: Why addition instead of multiplication?**

A: Addition allows both sources (current input and past memory) to contribute independently. Multiplication would create complex interactions that are harder to learn.

### Implementation

```python
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, x_t, h_prev):
        h_t = torch.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
        return h_t
```

---

## Why Vanilla RNNs Fail

### The Gradient Flow Problem

During backpropagation through time (BPTT), we compute:

```
∂hₜ/∂hₜ₋₁ = tanh'(...) · Wₕₕᵀ
```

For a sequence of length T:

```
∂hₜ/∂h₀ = [tanh'(...) · Wₕₕᵀ]ᵀ
```

The same matrix `Wₕₕ` gets multiplied T times!

### Mathematical Analysis

**Case 1: Exploding Gradients**
- If largest eigenvalue of `Wₕₕ` > 1
- After T multiplications: gradient ∝ λᵀ → ∞
- Network becomes unstable

**Case 2: Vanishing Gradients**
- If largest eigenvalue of `Wₕₕ` < 1  
- After T multiplications: gradient ∝ λᵀ → 0
- Network can't learn long-term dependencies

**Additional Factor**: `tanh'(x) ≤ 1` always
- This makes vanishing gradients even more likely
- After 20-30 timesteps, gradients effectively become zero

### Consequence
Vanilla RNNs cannot learn dependencies between events separated by more than ~10-20 timesteps.

---

## LSTM: The Solution

### The Key Insight

Instead of repeatedly multiplying by weight matrices, use **element-wise operations** with learned gates.

### Two Memory Systems

**1. Cell State (cₜ)**: The "memory highway"
- Information flows with minimal transformation
- Long-term memory storage

**2. Hidden State (hₜ)**: The "working memory"  
- What gets exposed to the next layer
- Short-term working memory

### The Gradient Flow Advantage

**Vanilla RNN**:
```
∂hₜ/∂hₜ₋₁ = tanh'(...) · Wₕₕᵀ    (involves matrix multiplication)
```

**LSTM**:
```
∂cₜ/∂cₜ₋₁ = fₜ                    (just element-wise multiplication!)
```

Where `fₜ` is the forget gate (values between 0 and 1).

**Why This Matters**:
- No repeated matrix multiplication
- If `fₜ ≈ 1`, gradient flows almost unchanged
- Can learn dependencies over 100+ timesteps

---

## LSTM Architecture Deep Dive

### The Three Gates + Candidate

An LSTM has four components at each timestep:

#### 1. Forget Gate (fₜ)
```
fₜ = σ(Wf @ [xₜ, hₜ₋₁] + bf)
```

**Purpose**: Decides what to keep from previous cell state
- fₜ = 1 → Remember everything
- fₜ = 0 → Forget everything  
- fₜ = 0.5 → Keep 50%

**Intuition**: "What old information is still relevant?"

#### 2. Input Gate (iₜ)
```
iₜ = σ(Wi @ [xₜ, hₜ₋₁] + bi)
```

**Purpose**: Decides how much new information to add
- iₜ = 1 → Add all new information
- iₜ = 0 → Ignore new information

**Intuition**: "How much of the new candidate should we store?"

#### 3. Candidate Cell State (g̃ₜ)
```
g̃ₜ = tanh(Wg @ [xₜ, hₜ₋₁] + bg)
```

**Purpose**: Creates new candidate values to add to cell state
- Uses tanh to squash values to [-1, 1]

**Intuition**: "What new information could we add to memory?"

#### 4. Output Gate (oₜ)
```
oₜ = σ(Wo @ [xₜ, hₜ₋₁] + bo)
```

**Purpose**: Decides what to output from cell state
- oₜ = 1 → Expose all of cell state
- oₜ = 0 → Hide cell state

**Intuition**: "What should we reveal from our memory?"

### The Update Equations

**Cell State Update**:
```
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ
```

Where ⊙ denotes element-wise multiplication.

**Breakdown**:
- `fₜ ⊙ cₜ₋₁`: Keep portion of old memory
- `iₜ ⊙ g̃ₜ`: Add portion of new candidate
- Result: Controlled update of long-term memory

**Hidden State Update**:
```
hₜ = oₜ ⊙ tanh(cₜ)
```

**Breakdown**:
- `tanh(cₜ)`: Squash cell state to [-1, 1]
- `oₜ ⊙ ...`: Filter what to expose
- Result: Working memory for next timestep and next layer

### Why This Design Works

1. **Additive Updates**: `cₜ = fₜ ⊙ cₜ₋₁ + ...` allows gradient to flow unchanged when fₜ = 1
2. **Learned Gates**: Network learns what to remember/forget/expose for each task
3. **Element-wise Operations**: No repeated matrix multiplications
4. **Separate Pathways**: Cell state for long-term memory, hidden state for short-term

---

## Implementation Details

### Efficiency Trick: Combined Weight Matrix

Instead of computing four separate transformations:
```python
fₜ = σ(Wf @ [xₜ, hₜ₋₁] + bf)
iₜ = σ(Wi @ [xₜ, hₜ₋₁] + bi)
g̃ₜ = tanh(Wg @ [xₜ, hₜ₋₁] + bg)
oₜ = σ(Wo @ [xₜ, hₜ₋₁] + bo)
```

We combine them into one big matrix multiplication:
```python
combined = [xₜ, hₜ₋₁] @ W + b        # One matrix multiply
fₜ, iₜ, g̃ₜ, oₜ = split(combined, 4)  # Split result
```

**Why?**
- One large matrix multiplication is much faster on GPUs than four small ones
- Reduces memory transfers
- Typical speedup: 3-4x

### Weight Matrix Dimensions

For an LSTM with `input_size=I` and `hidden_size=H`:

```
Input at each timestep: [xₜ, hₜ₋₁] has size (I + H)
Output (4 gates): each gate has size H, so total 4H

Weight matrix W: (I + H) × 4H
Bias vector b: 4H
```

### Initialization

Xavier/Glorot initialization:
```python
std = 1.0 / sqrt(hidden_size)
weights ~ Uniform(-std, std)
```

This helps prevent vanishing/exploding gradients at initialization.

---

## Complete Code

### LSTMCell Implementation

```python
import torch
import torch.nn as nn
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weight matrix for all 4 gates
        self.W = nn.Parameter(torch.randn(input_size + hidden_size, 4 * hidden_size))
        self.b = nn.Parameter(torch.randn(4 * hidden_size))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x_t, h_prev, c_prev):
        """
        Args:
            x_t: (batch_size, input_size) - current input
            h_prev: (batch_size, hidden_size) - previous hidden state
            c_prev: (batch_size, hidden_size) - previous cell state
        
        Returns:
            h_t: (batch_size, hidden_size) - new hidden state
            c_t: (batch_size, hidden_size) - new cell state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # Compute all gates in one matrix multiplication
        gates = combined @ self.W + self.b
        
        # Split into 4 gates
        f_t, i_t, g_t, o_t = gates.chunk(4, dim=1)
        
        # Apply activations
        f_t = torch.sigmoid(f_t)  # Forget gate
        i_t = torch.sigmoid(i_t)  # Input gate
        g_t = torch.tanh(g_t)     # Candidate
        o_t = torch.sigmoid(o_t)  # Output gate
        
        # Update cell state: c_t = f_t ⊙ c_prev + i_t ⊙ g_t
        c_t = f_t * c_prev + i_t * g_t
        
        # Update hidden state: h_t = o_t ⊙ tanh(c_t)
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t
```

### Complete LSTM Layer

```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create LSTM cells for each layer
        self.cells = nn.ModuleList([
            LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (seq_len, batch_size, input_size)
            hidden: tuple of (h_0, c_0) each (num_layers, batch_size, hidden_size)
        
        Returns:
            output: (seq_len, batch_size, hidden_size)
            (h_n, c_n): final hidden and cell states
        """
        seq_len, batch_size, _ = x.size()
        
        # Initialize hidden states if not provided
        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        else:
            h_0, c_0 = hidden
            h = [h_0[i] for i in range(self.num_layers)]
            c = [c_0[i] for i in range(self.num_layers)]
        
        outputs = []
        
        # Process each timestep
        for t in range(seq_len):
            x_t = x[t]
            
            # Pass through each layer
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = h[layer]  # Output becomes input to next layer
            
            outputs.append(h[-1])  # Store final layer output
        
        # Stack all outputs
        output = torch.stack(outputs, dim=0)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        
        return output, (h_n, c_n)
```

### Usage Example

```python
# Create 2-layer LSTM
lstm = LSTM(input_size=10, hidden_size=20, num_layers=2)

# Create input sequence: (seq_len, batch_size, input_size)
x = torch.randn(5, 3, 10)

# Forward pass
output, (h_n, c_n) = lstm(x)

print(f"Output shape: {output.shape}")        # (5, 3, 20)
print(f"Final h shape: {h_n.shape}")          # (2, 3, 20)
print(f"Final c shape: {c_n.shape}")          # (2, 3, 20)
```

---

## Key Takeaways

### 1. The Core Problem
Vanilla RNNs suffer from vanishing/exploding gradients due to repeated matrix multiplications during backpropagation through time.

### 2. LSTM's Solution
- **Cell state highway**: Information flows with minimal transformation
- **Learned gates**: Network decides what to remember/forget/expose
- **Element-wise operations**: Gradients flow more easily than through matrix multiplications

### 3. Mathematical Intuition

**Gradient Flow Comparison**:
```
Vanilla RNN: ∂hₜ/∂h₀ ∝ (Wₕₕᵀ)ᵀ → explodes or vanishes
LSTM:        ∂cₜ/∂c₀ = ∏ fᵢ     → stable when fᵢ ≈ 1
```

### 4. The Three Gates

- **Forget gate (fₜ)**: What to keep from old memory
- **Input gate (iₜ)**: How much new info to add  
- **Output gate (oₜ)**: What to expose from memory

### 5. Why It Works

The cell state update `cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ` creates an uninterrupted gradient highway when forget gate ≈ 1, allowing the network to learn dependencies over 100+ timesteps.

### 6. Layer Stacking

Multiple LSTM layers allow hierarchical learning:
- Layer 1: Learn basic patterns from raw input
- Layer 2: Learn complex patterns from Layer 1
- Layer N: Learn abstract representations

---

## Comparison Table

| Aspect | Vanilla RNN | LSTM |
|--------|-------------|------|
| Memory | Single hidden state | Cell state + Hidden state |
| Gradient Flow | Through Wₕₕ (unstable) | Through gates (stable) |
| Long-term Dependencies | ~10 timesteps | 100+ timesteps |
| Parameters | W_xh, W_hh, b | 4× more (4 gates) |
| Computation | tanh(Wx + Wh) | 4 gates + 2 states |
| Training Difficulty | Hard (vanishing gradients) | Easier (stable gradients) |