# Plan

Goal: build a ~1B-parameter LLM from scratch, inspired by nanochat, and ship it
with a ternary (1.57-bit) weight representation: signs in `{-1, 0, +1}` +
per-row fp16 magnitudes, ~2-bit packed on disk.

The work is broken into small, independently verifiable pieces. Each piece ends
with a concrete check that either passes or doesn't.

## Environment facts that shape the plan

- GPU: 1× RTX 4090 (24 GB), but a `llama-server` (27B model, serving this very
  agent) holds ~17.5 GB, leaving **~5.7 GB** for our work. Full 1B bf16 training
  needs ~16+ GB → the big run needs the server stopped or a rented GPU.
- CPU: i7-4790 (8 threads, no AVX-512), 31 GB RAM — fine for tokenization,
  small smoke tests, ternary quantization of the 1B checkpoint.
- Disk: ~586 GB free — plenty for data (ClimbMix shards are ~100 MB each).
- HuggingFace: plain `GET`/range requests to `karpathy/climbmix-400b-shuffle`
  work; no token needed. (Some `HEAD` probes return 401 — ignore them.)

## GPU status (2026-11-09)

llama-server (the 27B model serving this agent) holds 23.6 GB of the 24 GB
4090, and its KV cache grows with our conversation. **No GPU work is safely
possible right now.** Pieces 1-4 run on CPU; GPU pieces (5, 7, 8, 9-11) wait
for the user to stop/restart the server or provide another GPU.

## Pieces

### 1. Toolchain (mostly done)
- uv venv (Python 3.13) + torch (CUDA) + numpy/tiktoken/rustbpe/pyarrow. ✅
  (torch 2.15.0+cu132 imports, `torch.cuda.is_available()` is True)
- Remaining check: a bf16 matmul benchmark on the 4090 (~50 TFLOPs+).
  ✅ 2026-11-09: **163.2 TFLOPs** (8192³ bf16 matmul, 20 reps).

### 2. Model definition
- `model.py`: nanochat architecture (RoPE, QK-norm, GQA-ready, parameterless
  RMSNorm, no bias, relu² MLP, SSSL sliding windows, value embeddings, smear,
  backout, per-layer resid/x0 lambdas, logit softcap 15).
- Config presets: `tiny` (4L/256d, ~37M, CPU), `smoke` (8L/512d, ~126M, GPU),
  `tern1b` (21L/1408d, ~1.10B, target).
- Check: param counts match hand calculation; forward+backward runs on CPU for
  `tiny`; loss is finite and decreases a few steps of gradient descent.
  ✅ 2026-11-09 (CPU): param counts tiny 36.7M / smoke 125.8M / tern1b 1099.4M
  match D3; 5 AdamW steps on random tokens: loss 10.40 → 9.22; deterministic
  greedy `generate()` works; `setup_optimizer` (MuonAdamW) steps cleanly.
  (Found+fixed: `init_weights` crashed on layers without a value embedding.)

### 3. Tokenizer
- Train a 32768-vocab BPE (rustbpe, GPT-4-style split pattern) on ~1–2B chars
  of ClimbMix, same recipe as nanochat. Save pickle + token_bytes table.
- Check: encode/decode roundtrip on mixed text incl. unicode; ~5–6 tokens/word
  sanity on English.
  ✅ 2026-11-09: vocab 32768 trained on 1B ClimbMix chars + 6.4M-char
  symbolic slice (piece 13 insurance). ClimbMix 4.68 chars/token (nanochat
  ~4.8), symbolic 3.54 chars/token (not byte-shrapnel — slice worked),
  unicode/NUL/emoji roundtrips exact.

### 4. Data pipeline
- `dataset.py`: download N ClimbMix shards (val = last shard, as in nanochat).
- `tokenize.py`: tokenize to BOS-aligned best-fit packed rows (T+1 tokens),
  write uint32 `.bin` shards. ~35% token crop is expected (nanochat's number).
- `dataloader.py`: infinite streaming loader with state for resume.
- Check: `.bin` shard sizes match expected token count; dataloader yields
  correct shapes; first row starts with BOS.
  ✅ 2026-11-09: 10 train shards / 465M tokens + 46.5M val tokens, T=2048;
  all shards row-aligned, all first rows BOS; loader shift + resume
  exactness pass. (Found+fixed a real bug: next_batch counted CHUNKS not
  ROWS, yielding B(B+1)/2 rows per batch — would have silently poisoned
  every batch-size-dependent hyperparameter.)

### 5. Training loop
- `train.py` + `optimizer.py`: bf16 compute, fp32 master weights, Muon for 2D
  matrices + AdamW for embeddings/scalars (port of nanochat's single-GPU path),
  nanochat hyperparameters with muP-style LR scaling, batch/WD auto-scaling
  (Power Lines + T_epoch), cosine warmup/warmdown, val bpb every N steps,
  checkpoint save/resume, sample generation every N steps.
- Check: 100-step smoke run of the 126M model on the 4090 with ~4 GB budget;
  train loss decreases; val bpb finite; resume restores step + loss exactly.
  ✅ 2026-11-09 (run `smoke-bringup`): 100 steps, smoothed loss 10.40 →
  7.54; val bpb 3.15 (init) → 1.566; `--resume auto` restored step 100 and
  continued to 150 with smooth loss continuity (→ 6.54). Steady-state
  ~270k tok/s at B=262144 (16×8×2048). Fixed: cumulative tok/s metric was
  inflated after resume (windowed now). NOTE: real batch reached the
  model is correct ONLY after the dataloader chunk-count bug fix — any
  pre-fix run trained on 2.5× the intended tokens/step.

### 6. Ternary quantization
- `ternary.py`: per-row ternary (BitNet b1.58-style): threshold |w| per row,
  keep signs + per-row fp16 scale; 2-bit packed storage for signs.
  Two variants, both measured (see D9): the *standard* variant quantizes only
  transformer matrices (wte/lm_head stay bf16, value embeddings optional);
  the *max-compression* variant also quantizes all embedding tables.
  Quantizer instrumentation required: per-layer +/-/0 trit fractions, flip
  rate per step (for QAT), scale drift, export/reload agreement.
- Check: roundtrip `dequant(quant(W))` matches within expected error;
  ternary checkpoint of `smoke` model is ~4× smaller than bf16;
  val bpb delta vs bf16 measured and recorded in DECISIONS.md.
  ✅ partial 2026-11-09 (CPU): packing roundtrip EXACT (incl. non-multiple-
  of-4 widths); quantizer instrumentation prints per-layer trit fractions +
  rel err; both variants reload and generate. ✅ 2026-11-10: bpb delta
  measured on fp-control-126m checkpoint (503M tokens). See DECISIONS.md D13.

### 7. Inference
- `infer.py`: generate from a bf16 or ternary checkpoint (ternary is
  dequantized to bf16 at load; same forward code path). KV-cache-free naive
  decode is fine for a 1B model.
- Check: deterministic greedy decode on CPU for `tiny`; sane (not garbage)
  generation from the smoke model after a short training run.
  ✅ partial 2026-11-09: greedy decode on CPU works for bf16 and ternary-
  loaded checkpoints (`load_model_from_checkpoint`); open: "sane English"
  needs a trained model.

### 8. Small end-to-end
- Train `smoke` (126M) for a few hours on the 4090 alongside llama-server
  (fits in ~5 GB), ternarize, generate, record bf16-vs-ternary bpb delta.
- fp control training ✅ 2026-11-10: `fp-control-126m`, 1,920 steps / 503M
  tokens, final val bpb 0.965944 (falling across 8 evals), loss 3.18→3.01.
  Samples: mostly repetition at this token budget (expected for 126M).
- PTQ done: standard and max-compression variants written.
- Check: the ternary 126M model produces coherent English and the bpb delta is
  small (target: < 0.05 bpb). ❌ FAILED: bpb delta +0.66 (standard) / +0.82
  (max-compression); generation is repetition/gibberish. See DECISIONS.md D13.
  Expected for naive PTQ on an undertrained small model; QAT (piece 10) is the
  follow-up.

### 9. Big run (needs a decision)
- `tern1b` (1.10B): compute-optimal horizon is ~6.5B tokens
  (12 × scaling params). On the 4090 at ~10k tok/s that's ~6–9 days
  (corrected 2026-11-09, see D6); on a rented 8×H100 node it's < 1 day.
  User has approved multi-day exclusive GPU use (D11). Options:
  a) stop llama-server, bf16-train locally for N days (budget-limited),
     then PTQ (pieces 5-6),
  b) rent a cloud GPU,
  c) **TTF native-ternary training on the 4090** (new option from the
     research notes, see D9/D10): ~3 GB optimizer state, fits on 16 GB
     cards per the TTF README; the only option that may even fit *alongside*
     a restarted-but-leaner server,
  d) distill from a larger model (future work).
- Deliverable: trained 1.1B checkpoint (bf16 and/or ternary) + ternary
  packed checkpoint + bpb numbers + sample generations.

### 10. Matched QAT baseline (small scale, per research notes)
- Conventional ternary QAT on the `smoke` (126M) or `tiny` (37M) model:
  bf16 master weights + ternary fake-quant forward (STE), same data /
  tokenizer / arch / eval window as the fp control and the PTQ model.
- Variants: (i) inherited Muon/AdamW split, (ii) all-AdamW; (iii) quantize
  only matrices vs matrices+value_embeds (wte/lm_head always float in the
  standard variant).
- Check: ternary QAT beats PTQ at equal token budget (expected); quantizer
  instrumentation (trit fractions, flip rate, scale drift) recorded.
  ✅ 2026-11-10: `qat-126m-standard` 1,920 steps / 503M tokens, val bpb
  1.025228 vs fp 0.966 and PTQ 1.604. STE closes ~88% of the PTQ gap.
  Muon+STE stable; trit fractions stable at ~19%/61%/19%. See DECISIONS.md D16.

### 11. TTF small-scale experiment (experimental branch)
- Use `ttf-ternary-ref/` as the reference; run their trainer (or a ported
  core update) on a 37M/126M Llama-style model with matched data/tokens,
  against the piece-10 QAT baseline.
- Check: TTF reaches or beats master-weight QAT bpb at ≤ ~1 byte state per
  weight, at 126M scale, before any port into the main 1B run.

### 13. Curriculum ablation: symbolic-first pretraining (from neural-set-union)
- Idea (user, 2026-11-09): use the neural-set-union protocol (not necessarily
  its frozen circuits) — synthetic symbolic corpora (set algebra, logic,
  binding, function composition) as an early pretraining stage to nail the
  "linguistics of abstract reasoning" before general text, then specialize
  later (instruction following / coding / tool use).
- Arms at 126M, EQUAL total tokens: (a) symbolic stage -> ClimbMix,
  (b) symbolic mixed from the start (constant replay fraction, the standard
  anti-forgetting guard), (c) ClimbMix-only control,
  (d) **Anki arm** (user idea, 2026-11-09): symbolic stage, then ClimbMix
  with CLOSED-LOOP refresher replay — a small held-out symbolic probe is
  scored at every val window; when probe accuracy degrades past a
  predeclared threshold, inject refresher batches (target ~1-2% of stage-2
  tokens, decaying). Grounded in continual-learning replay results (small
  replay fractions prevent most forgetting) and maximal-interfered-
  retrieval (replay what is being forgotten); closed-loop scheduling at
  pretraining scale appears untested — novel and cheap.
- Winner decided on downstream val bpb + small
  reasoning probes AFTER stage 2 — never on symbolic accuracy alone (their
  pilot showed surface fluency without computation; reuse their shortcut
  controls: answer-frequency baselines, renaming/binding-change sensitivity).
  Secondary comparison: refresher-token spend of arm (d) vs arm (b)'s
  constant replay for equal retention.
- Sequencing constraint: generate the symbolic corpus (CPU, oracle-verified)
  BEFORE tokenizer training and include a slice in the BPE training data.
- Gate: runs after the piece-5/8 fp baseline exists (it IS the control arm);
  decision recorded before any 1B-scale use.

### 14. Multi-view (JEPA-style) auxiliary objective (after pipeline is proven)
- Source: Balestriero & LeCun, "Proving and Improving the Platonic
  Representation Hypothesis with JEPAs" (arXiv 2610.39302): JEPAs trained on
  GOOD VIEWS (two observations of one underlying state) reach representation
  alignment with 42x fewer samples than next-token prediction ("a JEPA with
  bad views"). The claim is about ALIGNMENT/sample-efficiency of
  representation learning, not directly downstream quality — our acceptance
  metric stays downstream (val bpb + probes); alignment/effective-rank are
  diagnostics only (constant vectors align perfectly and are useless).
- Text-only entry: no images. The piece-13 symbolic generator emits paired
  views for free — a problem as a symbolic expression AND as independent
  prose are two renderings of one semantic object with oracle-verified
  information preservation. Attach a projection head to the decoder's
  problem representation; predict view-B's (EMA, stop-grad) representation
  from view-A; explicit anti-collapse regularizer chosen BEFORE the run.
  Next-token loss continues unchanged; no paired view or teacher at test
  time; alignment targets must not contain answer information (no leak).
- Factorial control (from neural-set-union's MULTIVIEW.md): with/without
  the alignment loss, BOTH seeing the same paired views — presence of extra
  descriptions must not be mistaken for the loss's benefit. Match projection
  capacity, count teacher/predictor FLOPs in the cost.
- Gate: after pieces 13 (needs its generator) and 5/8 (fp control); run at
  126M on the FP control first (one-change-at-a-time, per USER_NOTES);
  only a win gets ported into ternary runs.

### 12. Post-training: distillation + alignment ("dirty train")
- Sequence-level distill from a license-permitting teacher (Qwen/DeepSeek/
  Kimi family — verify the specific model license when chosen; Qwen3 is
  Apache-2.0): generate with teacher, re-tokenize with our BPE, SFT the
  ternary 1B (dequantized-forward QAT-continue or direct finetune — decide
  from piece 10/11 results). Light preference optimization afterwards.
- Teacher inference can run on this same 4090 between training windows.
- Check: benchmark vs Qwen3.5-0.9B (D11) on agreed small evals + bpb;
  quality-per-byte recorded in DECISIONS.md.

## Operating rhythm (D11)

- pi runs standup and spot-checks every N training cycles; Qwen manages
  routine test runs during agent phases. GPU exchange via
  `scripts/swap.sh` + `scripts/handoff.sh` only.
- Every session ends with the insights question and an `INSIGHTS.md` entry
  BEFORE handoff to training (handoff.sh appends the dated template).

## Verification standard

Every piece has a check. A piece is "done" when the check passes and the
command that reproduces it is in this file or the Makefile.
