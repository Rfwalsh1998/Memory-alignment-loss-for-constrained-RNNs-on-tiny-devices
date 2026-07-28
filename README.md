# Exact State Carry

**KV-Free Geometric State Propagation**

A tiny experimental recurrent language model whose normalized hidden state is also its prediction vector.

There is no embedding table, attention mechanism, KV cache, output projection, or learned logits head. The state dimension equals the vocabulary size, and each coordinate corresponds directly to one vocabulary item.

This project began as a roast of an intentionally minimal architecture. The roast became less convincing after the model reached roughly 48% held-out next-character accuracy on the raw 77-character KJV corpus after about 200 seconds of single-threaded CPU training.

## Core update

For one-hot input \(x_t\) and state \(h_{t-1}\),

\[
g_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
\]

\[
c_t = \tanh(W_i x_t + U_i h_{t-1} + b_i)
\]

\[
q_t = h_{t-1} + g_t \odot c_t
\]

\[
h_t = \frac{q_t}{\lVert q_t\rVert_2}
\]

The one-hot next token is \(z_t\). Training minimizes

\[
\mathcal L_t = 1 - \cos(z_t,h_t).
\]

Because both vectors are unit length and \(z_t\) is one-hot, the cosine is exactly the target coordinate of \(h_t\).

## Properties

- Constant-size recurrent inference state: \(O(|V|)\), independent of sequence length.
- Hidden state dimension equals vocabulary size.
- The state is normalized after every recurrent update.
- Numerical state is carried between contiguous chunks.
- The gradient graph is truncated at chunk boundaries.
- Prediction is read directly from state coordinates.
- The included sampler clamps negative coordinates to zero, keeps the top \(k\), and normalizes those values. This is an unlearned sampling rule, not a projection head.

## Included KJV checkpoint

The included checkpoint was trained on a contiguous 95% split of the raw supplied KJV text. The final 5% was held out.

| Measurement | Result |
|---|---:|
| State/vocabulary size | 77 |
| Learned parameters | 23,870 |
| CPU threads | 1 |
| Accumulated training time | 200 seconds |
| Optimization steps | 4,154 |
| Contiguous passes recorded | 7 |
| Held-out cosine loss | 0.49083 |
| Held-out argmax character accuracy | 48.35% |
| Maximum state-norm error | \(3.58\times10^{-7}\) |

The 180-second evaluation reached 48.56%, so the small change at 200 seconds is ordinary checkpoint/evaluation fluctuation rather than a monotonic accuracy claim.

The autonomous samples remain noisy pseudo-KJV. Teacher-forced next-character accuracy is the meaningful result here.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Evaluate the bundled checkpoint and generate from a common multiword prompt:

```bash
python scripts/train_kjv.py
```

Continue training it for another 20 seconds:

```bash
python scripts/train_kjv.py --train-seconds 20
```

Start over from random initialization:

```bash
python scripts/train_kjv.py --fresh --train-seconds 120
```

Use a different prompt:

```bash
python scripts/train_kjv.py \
  --prompt "And David said unto Saul, Let no man's heart fail because of him; " \
  --generate-length 800
```

Run the synthetic delayed-key benchmark:

```bash
python scripts/delayed_recall.py --train-seconds 120
```

## Files

- `exact_state_carry/model.py` — reusable model and exact cosine loss.
- `scripts/train_kjv.py` — raw KJV training, checkpoint resume, evaluation, and generation.
- `scripts/delayed_recall.py` — synthetic long-delay key benchmark.
- `data/kjv.txt` — the supplied public-domain training corpus.
- `checkpoints/kjv_exact_state_200s.pt` — the included trained checkpoint.
- `MODEL_CARD.md` — scope, metrics, and limitations.

## Important limitations

This is an experiment, not evidence that a 77-dimensional recurrent state can replace a general language model or a KV cache in every setting.

Normalization fixes state magnitude, but it does not by itself prove that gradients cannot vanish, explode, become ill-conditioned, or lose information.

The KJV result measures next-character prediction on one corpus with one contiguous split. It does not establish broad language understanding, robust long-context recall, or competitive perplexity.

The checkpoint is a Python/PyTorch serialized file. Load checkpoints only from sources you trust.

## Public domain

The software and included checkpoint are released under the Unlicense. The bundled KJV text is supplied as public-domain source material according to its header.

See `LICENSE`.
