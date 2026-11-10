# Social post drafts (edit to taste before posting)

## X / Twitter

Ternary-weight LLM trained from scratch on one RTX 4090. Matched fp/QAT/PTQ
runs, 126M params, 503M tokens, ~1.2 GPU-hours each:

• QAT lands within +0.053 bpb of the fp model
• fully-ternary variant deploys at 61 MB vs 252 MB (-76%) for +0.093 bpb
• naive PTQ: catastrophic (+0.66 bpb)

Full pipeline + honest numbers: <github link>

## LinkedIn

We trained a 1.57-bit (ternary) language model from scratch on a single
consumer GPU — and measured everything honestly.

Three matched runs (126M params, identical data/tokenizer/503M-token
budget): full bf16, naive post-training ternarization, and
quantization-aware training from random init. QAT comes within +0.053
bits-per-byte of the fp model; the max-compression ternary model deploys at
61 MB — 4× smaller than the fp model — for under a tenth of a bpb.

The whole pipeline is open source (MIT): nanochat-style architecture, Muon
optimizer, per-row BitNet-style ternarization, and a one-command repro at
~1.2 GPU-hours per arm. Writeup + code: <github link>

## Hacker News (Show HN)

Title: Show HN: Ternary-weight LLM trained from scratch on one RTX 4090

Text:
We built a from-scratch training pipeline (nanochat-style arch, Muon
optimizer, 32k BPE, packed ClimbMix data) and ran matched fp / PTQ / QAT
experiments at 126M params / 503M tokens on a single 4090 (~1.2h per run).

QAT from random init lands within +0.053 bpb of the fp control; the
fully-ternarized model deploys at 61 MB vs 252 MB for +0.093 bpb. Naive
PTQ is catastrophic (+0.66), which is why the matched-baseline discipline
matters. Everything is MIT-licensed with honest byte accounting and a full
decision log, including the mistakes (see DECISIONS.md and INSIGHTS.md).

Repo: <github link>
