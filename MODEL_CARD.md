# Model Card: Exact State Carry — KJV 200s

## Model

A recurrent neural network with a 77-dimensional hidden state tied directly to a 77-character vocabulary.

Trainable components:

- \(W_f\), \(U_f\), \(b_f\)
- \(W_i\), \(U_i\), \(b_i\)

Total trainable parameters: **23,870**.

The model has no trainable embedding table, attention mechanism, KV cache, output projection, logits head, or auxiliary classifier.

## Data

The included raw KJV text contains **4,602,958 encoded characters** after consuming the UTF-8 byte-order marker as file metadata. The remaining raw text has exactly **77 unique characters**.

Training uses the first 95% of the contiguous character stream. Validation uses the final 5%.

This split tests unseen text from the same corpus and style. It is not an independent-domain evaluation.

## Training configuration

- State size: 77
- Sequence length: 128
- Batch size: 64
- Optimizer: Adam
- Learning rate: \(2\times10^{-3}\)
- Gradient clipping: 1.0
- Numerical state carried between chunks
- Gradient graph detached at every 128-character boundary
- Single CPU thread
- Accumulated checkpoint training time: 200 seconds

## Evaluation

Using 32 contiguous validation chunks with carried state:

- Held-out cosine loss: **0.49083**
- Held-out argmax character accuracy: **48.35%**
- Maximum unit-norm error: **\(3.58\times10^{-7}\)**

At 180 seconds, the same protocol measured 48.56% accuracy and 0.49154 loss. The 200-second checkpoint improved the continuous loss slightly while argmax accuracy fluctuated downward by about 0.25 percentage points.

## Generation

Generation does not use a learned decoder.

The sampler:

1. reads the state coordinates directly,
2. clamps negative coordinates to zero,
3. retains the top six positive coordinates,
4. normalizes them into sampling weights,
5. samples one character,
6. feeds that character back into the recurrence.

Generated text is locally KJV-like but remains largely incoherent at this training stage.

## Intended use

- Recurrent-state geometry experiments
- Constant-memory sequence modeling research
- Direct latent prediction experiments
- Delayed-recall benchmarks
- Educational inspection of hyperspherical recurrent dynamics

## Not established by this checkpoint

- General language-model competence
- Reliable factual generation
- Competitive perplexity
- Guaranteed thousand-token recall
- Freedom from vanishing/exploding gradients
- A universal replacement for attention or KV caches

## License

Software and checkpoint: Unlicense/public-domain dedication.

Corpus: supplied as public-domain KJV text according to the file header.
