# nanotern

**Ternary-weight (1.57-bit) language models trained from scratch on a single
consumer GPU** — a from-scratch pipeline inspired by
[karpathy/nanochat](https://github.com/karpathy/nanochat), with matched
fp / PTQ / QAT experiments and honest quality-per-byte accounting.

Headline result (126M params, 503M tokens, one RTX 4090, ~1.2 h/run):

| model | val bpb | Δ vs fp | deployed size |
|---|---|---|---|
| fp control | 0.9450 | — | 251.7 MB |
| PTQ (naive) | 1.6040 | +0.659 | — |
| **QAT standard** | 0.9977 | **+0.053** | 207.7 MB |
| **QAT max-compression** | 1.0378 | **+0.093** | **61.2 MB** |

Full writeup, caveats, and reproduction commands:
[**REPLICATION_REPORT.md**](REPLICATION_REPORT.md).
Every design decision and why: [DECISIONS.md](DECISIONS.md).
Work plan with per-piece verification checks: [PLAN.md](PLAN.md).

## Quickstart

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

python -m nanotern.dataset -n 12                 # download ClimbMix shards
python -m nanotern.tok_train --max-chars 1000000000   # train 32k BPE (~2 min)
python -m nanotern.tokenize --split train --n-shards 10
python -m nanotern.tokenize --split val

python -m nanotern.train --config smoke --run-name fp-control       # fp baseline
python -m nanotern.train --config smoke --run-name qat --qat        # ternary QAT
python eval_bpb.py --checkpoint ~/.cache/nanotern/checkpoints/qat/step_0001920.pt
python -m nanotern.ternary --from-checkpoint <ckpt> --out tern.pt   # PTQ export
python -m nanotern.infer --checkpoint tern.pt --prompt "Hello"      # generate
```

`--config tiny` runs end-to-end on CPU for a smoke test (37M params).

## What's here

| file | purpose |
|---|---|
| `nanotern/model.py` | nanochat-style GPT (RoPE, QK-norm, relu² MLP, value embeddings, SWA) |
| `nanotern/optimizer.py` | Muon + AdamW fused single-GPU optimizer |
| `nanotern/qat.py` | TernaryLinear: per-row ternary forward + STE |
| `nanotern/ternary.py` | PTQ quantizer, 2-bit sign packing, honest byte accounting |
| `nanotern/train.py` | training loop (nanochat hyperparameter scaling, cosine schedule, resume) |
| `nanotern/dataset.py` `tokenize.py` `dataloader.py` | ClimbMix → packed .bin pipeline |
| `eval_bpb.py` | bits-per-byte eval on a fixed val window |

## Status & roadmap

Done: matched 126M fp/PTQ/QAT experiments (above). In flight: seed
replicates, all-AdamW ablation, master-weight-free TTF training (piece 11),
then a ~1B run on the same 4090. See PLAN.md pieces 9–14.

## License

MIT (code). ClimbMix data and model checkpoints are subject to their own
terms; no checkpoints are distributed in this repo.
