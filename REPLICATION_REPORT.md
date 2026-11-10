# Training a ternary-weight LLM from scratch on one RTX 4090

*A matched fp / PTQ / QAT comparison at 126M parameters, with honest numbers.*

**TL;DR** — We trained a 126M-parameter transformer (nanochat-style
architecture) from scratch on one consumer GPU, three ways: full bf16,
naive post-training ternarization (PTQ), and quantization-aware training
(QAT) with ternary weights from random init. QAT nearly matches the fp
model (**+0.053 bpb**) while the fully-ternarized "max-compression" variant
deploys at **61 MB vs the fp model's 252 MB (−76%)** for +0.093 bpb.
Naive PTQ is catastrophic (+0.66 bpb). All runs are matched: same data,
same tokenizer, same 503M-token budget, same evaluation window. Everything
to reproduce this is in this repo; each run is ~1.2 GPU-hours.

## Why

Two questions drove this:

1. Can you do *real* ternary-training research on a single consumer GPU?
   (A 24 GB RTX 4090, in a personal workstation, total cost of these
   experiments: an afternoon of electricity.)
2. What does ternary quantization actually cost when you measure it
   honestly — equal token budgets, equal evaluation, deployed bytes
   counted including everything that stays floating point?

The weight format is BitNet b1.58-style ternary: every weight in
{-1, 0, +1} with a per-output-row fp16 scale, signs packed 2 bits each.
"1.57-bit" ≈ log₂(3).

## Setup

- **Model**: 126M-param decoder-only transformer, faithful
  reimplementation of the nanochat architecture (RoPE, QK-norm, relu² MLP,
  value embeddings, sliding-window attention). Config `smoke`: 8 layers,
  512 dim.
- **Data**: ClimbMix (`karpathy/climbmix-400b-shuffle`), pre-tokenized
  into packed 2048-token rows; validation is the held-out last shard.
- **Tokenizer**: 32,768-vocab BPE trained on 1B chars of the training
  distribution (rustbpe + tiktoken), 4.68 chars/token on English.
- **Training**: Muon (matrices) + AdamW (embeddings/scalars), nanochat's
  hyperparameter scaling system, 503M tokens = 12× the matmul parameter
  count (compute-optimal by the nanochat rule of thumb), 1,920 steps of
  262,144 tokens, ~1.2 h per run on the 4090 (~120k tokens/s).
- **Quantizer**: per-row threshold `t = mean(|row|)`, sign ±1 outside, 0
  inside; row scale `‖row‖₂ / √nnz`. QAT uses the same quantizer in the
  forward pass with a straight-through estimator; master weights stay
  floating point during training. Two variants: **standard** (transformer
  matrices ternary; embeddings/head float) and **max-compression**
  (embedding tables ternary too; lm_head stays float).
- **Metric**: bits-per-byte (bpb) on a fixed 2M-token window at the start
  of the validation shard. Deployed bytes count packed signs + fp16 scales
  + all float tensors in bf16. No cherry-picked generations.

## Results

| model | val bpb | Δ vs fp | deployed size | vs fp size |
|---|---|---|---|---|
| fp control (bf16) | 0.9450 | — | 251.7 MB | 1× |
| PTQ standard | 1.6040 | +0.659 | — | — |
| PTQ max-compression | 1.7644 | +0.819 | — | — |
| **QAT standard** | **0.9977** | **+0.053** | 207.7 MB | 0.83× |
| **QAT max-compression** | **1.0378** | **+0.093** | **61.2 MB** | **0.24×** |

- QAT closes **92%** of the PTQ gap (standard) and **89%** (max-comp).
- Max-compression pays only +0.040 bpb over standard QAT and is **3.4×
  smaller** — the clear quality-per-byte winner.
- QAT quantizer telemetry was healthy throughout: per-layer trit fractions
  steady at ~19% / 62% / 19% (−/0/+), no divergence, no Muon+STE
  instability, ~4% throughput overhead vs the fp run.

## What this does and doesn't establish

**Does:** at matched tokens/data/eval, ternary QAT from random init is
viable on commodity hardware and lands within ~0.05–0.09 bpb of its fp
twin; naive PTQ of a model this small/undertrained is not viable;
quality-per-byte strongly favors ternarizing embeddings too.

**Doesn't (yet):** these are single-seed runs at 126M params / 503M tokens
(seed-2 replicates are in flight; the repo will be updated). This is not a
claim about 1B+ scale, about downstream-task quality (bpb only), or about
master-weight-free training (that's the TTF experiment, next on our
queue). The fp model itself is a research-grade 126M, not a chat model.

## Reproduce

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
python -m nanotern.dataset -n 12                 # ~1.2 GB ClimbMix shards
python -m nanotern.tok_train --max-chars 1000000000
python -m nanotern.tokenize --split train --n-shards 10
python -m nanotern.tokenize --split val
python -m nanotern.train --config smoke --run-name fp-control            # fp arm
python -m nanotern.train --config smoke --run-name qat-std --qat         # QAT arm
python -m nanotern.train --config smoke --run-name qat-max --qat \
    --qat-variant max-compression                                        # max-comp arm
python eval_bpb.py --checkpoint <ckpt.pt>                                # eval any arm
python -m nanotern.ternary --from-checkpoint <ckpt.pt> --out tern.pt     # PTQ export
```

Full work log with every decision and its rationale: [`DECISIONS.md`](DECISIONS.md).
Things that occurred to the agents along the way: [`INSIGHTS.md`](INSIGHTS.md).

## Credits / provenance

- Architecture and training recipe: reimplemented from
  [karpathy/nanochat](https://github.com/karpathy/nanochat) (MIT) as a
  spec; no nanochat code is imported.
- Quantizer: BitNet b1.58-style per-row ternarization.
- Muon optimizer: Keller Jordan's modded-nanogpt lineage (Polar Express /
  NorMuon variants), ported single-GPU.
- This project is developed in the open by a human + LLM agents (the
  commit log is honest about which commits are agent-authored).
