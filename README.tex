# Exact State Carry: KV-Free Geometric State Propagation

## Overview
This repository implements an experimental recurrent neural network (RNN) architecture that achieves long-horizon memory without Attention mechanisms or a Key-Value (KV) cache. 

Instead of relying on an expanding memory footprint to recall past tokens, it relies on **geometric state propagation** on a unit hypersphere. By tying the hidden state dimension directly to the vocabulary size and optimizing via a direct cosine loss, the model learns to "carry" vital information (like keys or identifiers) across thousands of tokens of noise using constant $O(1)$ memory.

---

## The Core Innovation: Direct Latent Prediction

Standard language models use a hidden state $\mathbf{h}_t$ which is then projected through a massive dense layer ($\mathbf{W} \mathbf{h}_t$) to generate logits. This model removes that projection layer entirely. 

The hidden state dimension ($N=77$) is strictly equal to the vocabulary size. **The hidden state *is* the prediction.** 

To train it, we force the hidden state vector to geometrically align with the one-hot encoded target token using a pure cosine similarity loss:

$$ \mathcal{L}(t) = 1 - \frac{\mathbf{z}_{seq}(t) \cdot \left( \frac{\mathbf{h}_{t-1} + \sigma\Big( \mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f \Big) \odot \tanh\Big( \mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i \Big)}{\left\| \mathbf{h}_{t-1} + \sigma\Big( \mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f \Big) \odot \tanh\Big( \mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i \Big) \right\|_2} \right)}{\|\mathbf{z}_{seq}(t)\|_2} $$

Where:
*   $\mathbf{z}_{seq}(t)$ is the one-hot target token vector.
*   The complex denominator guarantees the updated state is L2-normalized.
*   $\sigma(\dots)$ acts as a gating mechanism to determine what new information to add to the existing state.

---

## Why This Matters

### 1. KV-Free Constant Memory
Transformers suffer from $O(N)$ or $O(N^2)$ memory scaling during inference because they must store past keys and values to attend to them later. By projecting the memory entirely into the geometric position of a single $N$-dimensional state vector, memory utilization remains $O(1)$ regardless of sequence length.

### 2. Hyperspherical Stability (No Exploding Gradients)
Traditional RNNs struggle with long contexts because repeated matrix multiplications cause gradients to vanish or explode. This architecture solves that by treating the recurrent update as vector addition, immediately followed by L2 normalization:

$$ \mathbf{q}_t = \mathbf{h}_{t-1} + \text{gated\_input} $$
$$ \mathbf{h}_t = \frac{\mathbf{q}_t}{\|\mathbf{q}_t\|_2} $$

Because the state is perpetually clamped to the surface of a unit hypersphere, the magnitude never grows. The network learns to navigate the sequence by rotating the state vector rather than scaling it.

### 3. Orthogonal Memory Carry
By lacking a linear projection layer at the end, the network is forced to learn an inherently semantic geometry. To remember a token from 1,000 steps ago, it must isolate that information in orthogonal dimensions on the hypersphere, shielding it from being overwritten by the "noise" of current token predictions.

---

## Implementation Details & Mechanics

The PyTorch implementation (`ExactStateCarry`) relies on several specific design choices to make this geometric propagation work:

*   **Residual Update over Matrix Multiply:** Notice that the update is `h_prev + (forget_gate * input_content)`. We do *not* multiply `h_prev` by a recurrent weight matrix for the core state carry. The recurrence is pure addition, preventing vanishing gradients.
*   **Target Coordinate Maximization:** The `exact_cosine_loss` bypasses `CrossEntropyLoss`. It creates a one-hot vector of the target, L2 normalizes the state predictions, and maximizes the cosine similarity. The model learns to point the state vector precisely at the coordinate corresponding to the next token's index.
*   **Orthogonal Initialization:** The recurrent weights (`U_i`) are initialized orthogonally (`nn.init.orthogonal_`). This is critical for early training stability, as it preserves vector norms during the initial linear transformations before the non-linearities and L2 clamp take over.
*   **Continuous Batching:** The `StreamBatcher` guarantees that token $N+1$ in a batch is the exact continuation of token $N$ from the previous chunk. The hidden state `h_prev` is `detach()`ed and carried forward infinitely across batch boundaries (Truncated Backpropagation Through Time). This allows the model to learn dependencies much longer than the `SEQ_LEN=128` window.

---

## The Benchmark: Delayed Recall

To prove the architecture's long-horizon retention, this codebase includes a synthetic "needle-in-a-haystack" benchmark.

The model is fed a sequence like:
`packet[4021] zibble mork talven quux ... [50 filler words] ... return[4021]`

The network must recall the exact 4-digit key (`4021`) after processing hundreds of characters of completely random synthetic trash words. Because it lacks a KV cache, it cannot "look back" at the packet ID. It must encode the key into its 77-dimensional hidden state, continuously update that state to predict the filler words, and then successfully decode the original key from the surviving geometry of the vector when prompted by the `return[` string.

---

## Quick Start & What to Do Next

### Running the Code
The script is self-contained and runs on the CPU by default. 
1. Ensure you have PyTorch installed (`pip install torch`).
2. Run the script: `python train.py`

The script streams JSON telemetry to `stdout`, detailing validation loss, accuracy, and the crucial `delayed_key_accuracy` metric. After 120 seconds, it runs a retention probe and generates a synthetic text sample.
