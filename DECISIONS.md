# Decisions

Important decisions with rationale. Oldest first; D11 is the latest.

## D1. "1.57-bit" means ternary weights (BitNet b1.58-style)

2^1.57 ≈ 2.97, and log₂(3) ≈ 1.585 bits. The practical interpretation is
ternary weights `{-1, 0, +1}` with a per-row (per-output-channel) fp16
magnitude. This is exactly the BitNet b1.58 representation
("b1.58" = log₂3 bits per weight).

Chosen quantizer (post-training, per row of each 2D weight/embedding matrix):
- threshold `t = mean(|row|)` (alternatives tried: median, 0.5·max — recorded
  in piece 6 results)
- sign: `+1` if `w > t`, `-1` if `w < -t`, else `0`
- scale `s = ||row||₂ / sqrt(nnz(row))`
- dequant: `W ≈ s · sign(W)`

Storage: signs packed 2-per-byte (uint8), scales fp16 per row. For the 1.10B
model: ~275 MB total checkpoint (vs ~2.2 GB bf16, ~4.4 GB fp32).

Why post-training quantization (PTQ) first, not quantization-aware training
(QAT): PTQ is a single CPU script, gives the bpb-delta number fast, and
dequantizing to bf16 makes the inference path identical to the bf16 model.
If the delta is too large, QAT (straight-through estimator on the ternary
sign) is the follow-up — the code structure (ternary.py) supports it.

## D2. Architecture: faithful port of nanochat's GPT, single-GPU

We follow nanochat's architecture rather than a generic "Llama-style 1B":
RoPE (base 100000), QK-norm (×1.2 scale), no biases, parameterless RMSNorm,
relu² MLP (4× width), untied wte/lm_head, value embeddings on alternating
layers (ResFormer, input-gated), smear (previous-embedding mixing),
backout (mid-layer residual subtraction, λ=0.2), per-layer resid_lambdas
(1.15→1.05) and x0_lambdas (0.20→0.05), logit softcap 15, SSSL sliding-window
attention (S = quarter context, 768 at T=2048; final layer always full).

Reason: nanochat is the reference the user pointed at; its hyperparameters
(LRs, schedules, inits) are tuned for exactly this architecture family and
transfer via muP-style scaling (LR ∝ 1/√(dmodel/768)). Reusing the recipe
removes a whole class of "my 1B is bad because my hyperparameters are wrong"
risk. Everything is reimplemented from scratch in our own files (we read
nanochat's source as the specification), single-GPU only (no DDP paths).

Attention: nanochat uses FA3 on H100, SDPA fallback elsewhere. We use the
SDPA path directly (works on the 4090, no extra packages): `is_causal=True`
(flash kernel) for full-context layers, explicit bool mask for
sliding-window layers. Memory cost of the S-layer mask at T=2048 is fine for
our batch sizes.

## D3. Model size: 21 layers × 1408 dim ≈ 1.10B params

nanochat's depth dial is dmodel = depth × 64 rounded up to a multiple of
head_dim 128. At depth 21: dmodel = 1408, n_head = 11 (head_dim 128).

| group | params |
|---|---|
| wte + lm_head (untied) | 2 × 32768 × 1408 = 92.28M |
| value_embeds (11 of 21 layers) | 11 × 32768 × 1408 = 507.52M |
| transformer matrices (21 × [qkv+proj+c_fc+c_proj]) | 499.84M |
| norms + scalars | ~0.09M |
| **total** | **~1.100B** |

Note: value embeddings are ~46% of all parameters. This is a nanochat
property (strong small-model recipe) and is fine for ternary: they are just
lookup tables quantized per token-row. Smaller presets for smoke tests:
`smoke` = 8L/512d ≈ 126M (GPT-2 class), `tiny` = 4L/256d ≈ 37M (CPU).

## D4. Data: ClimbMix (karpathy/climbmix-400b-shuffle), pre-tokenized to .bin

- nanochat trains on `karpathy/climbmix-400b-shuffle` (parquet shards,
  `text` column, ~250M chars / ~100 MB each; last shard is validation).
  We verified from this machine that plain GET/range requests to the shards
  work without authentication.
- nanochat tokenizes on the fly from parquet. We instead pre-tokenize to
  packed uint32 `.bin` shards (classic nanogpt layout: rows of T+1 tokens,
  BOS-aligned best-fit packing, 100% utilization, ~35% crop). Reason:
  training I/O becomes a simple `mmap` read (no parquet/tiktoken in the
  hot loop), resume is trivial (byte offset), and tokenization cost is
  paid once.
- Tokenizer: we train our own BPE (rustbpe + tiktoken, GPT-4-style split
  pattern, 32768 vocab) on ~1B chars, same as nanochat. Reason: nanochat's
  hyperparameters are tuned to its tokenizer's ~bytes/token; using an
  off-the-shelf tokenizer with different statistics would invalidate the
  inherited batch-size/LR schedule reasoning.

## D5. Optimizer: Muon for matrices + AdamW for the rest (port)

Port of nanochat's `MuonAdamW` fused-kernel optimizer, single-GPU path only.
Muon (momentum + Polar-Express orthogonalization + NorMuon variance
reduction + cautious weight decay) for all 2D matrix params; AdamW for
wte/lm_head/value_embeds/scalars. Hyperparameters from nanochat defaults:
embedding_lr 0.3, unembedding_lr 0.008, matrix_lr 0.02, scalar_lr 0.5,
weight_decay 0.28, muP scaling 1/√(dmodel/768), batch-LR scaling
√(B/524288), WD scaling per T_epoch.

## D6. Training regime for tern1b (pending user decision on budget)

- T = 2048, bf16 compute, fp32 master weights (no fp8 — 4090 only).
- Compute-optimal horizon: 12 × (matrices + lm_head) = 12 × 546M ≈ 6.55B
  tokens → auto batch ≈ 2²⁰ = 1,048,576 tokens (≈ 16 grad-accum steps of
  32×2048 on the 4090).
- 2026-11-09: user approved multi-day exclusive GPU use (server down is
  fine); compute-optimal horizon is the default plan for the big run.
- 4090 throughput estimate: 6×1.1e9 FLOPs/token = 6.6e9, at ~70–90 TFLOPs
  effective (40-55% MFU with SDPA) → ~10–13k tokens/s → 6.55B tokens ≈ 6–8
  days of exclusive GPU. (Fixed 2026-11-09: this line previously said
  "6.6e15 FLOPs/token → ~100-130 tok/s → 14-18 days"; the e15 was a units
  error and the tok/s figure was off by ~100×. True throughput measured in
  piece 1/5 before committing the big run.)
  Budget-limited runs (e.g. 500M–1B tokens, ~0.5–1 day) will be
  undertrained but give a usable ternary artifact + honest bpb numbers.

## D7. Inference: dequantize-to-bf16 first

v1 ternary inference dequantizes to bf16 at load and runs the standard
forward pass (simple, correct, same code path as bf16). A fused ternary GEMM
(signs × magnitudes in int8-ish arithmetic) is a later optimization, not a
correctness requirement — at 1B params the dequantized model still fits
comfortably in ~2.4 GB bf16 and decodes fast.

## D8. Project structure & tooling

- Own codebase in `nanotern/`, `nanochat-ref/` kept as read-only reference.
  We reimplement; we do not import nanochat.
- Python 3.13 venv via uv (system python is 3.14, torch wheels safest on 3.13).
- Cache dir `~/.cache/nanotern/` (data/, tokenizer/, checkpoints/) — mirrors
  nanochat's `NANOCHAT_BASE_DIR` convention.
- No wandb; plain log file + printed metrics.

## D9. Research context (per USER_NOTES_TO_QWEN.md, read 2026-11-09)

The user pointed at recent work on ternary LLMs. References cloned/fetched:
- `ttf-ternary-ref/` (oguzeldereli/ttf-ternary) — TTF trainer + paper
  (NOTE 2026-11-09: not actually cloned into the repo yet — piece 11 must
  start with `git clone https://github.com/oguzeldereli/ttf-ternary ttf-ternary-ref`.
  `nanochat-ref/` exists at /home/albert/git/nanochat-ref, outside this repo.)
- `/tmp/ttf.pdf` — "Training Ternary LLMs Without Master Weights" (UCL, Oct 2026)
- Axolotl/TII blog (Apr 2026) — TII BitNet fine-tuning, `onebitllms` conversion
- Spectra (ICLR 2025), TriTera (ACL 2025), Hestia (arXiv 2601.20745),
  CAT-Q (arXiv 2606.26650), BITCOS (arXiv 2609.16338) — cited in notes

How this changes the plan:

1. **Goal reframing.** The valuable deliverable is a *reproducible ternary
   training recipe with honest quality-per-byte and quality-per-GPU-hour
   numbers*, not "a competitive 1B assistant on one 4090". Small matched
   experiments (fp control vs ternary, identical data/tokenizer/arch)
   beat one big opaque run.
2. **PTQ-first stays** (D1) — it is the conventional baseline the notes
   tell us to establish first. Changes are then tested one at a time.
3. **TTF is the headline experimental branch.** TTF (two-timescale threshold
   flips) keeps weights packed as trits during training, ~0.6-0.7 state
   numbers/weight (≈3 GB for 1.1B) vs 12+ bytes for master weights. Their
   README: "a 16 GB card trains up to ~1B parameters directly". This is the
   only plausible path to *training the 1B model on this 4090* (see D10).
   Per the notes: reproduce a small matched experiment before porting.
   New pieces added to PLAN.md: matched QAT baseline (piece 10) and TTF
   small-scale experiment (piece 11).
4. **Optimizer audit.** Do not assume nanochat's Muon/AdamW split is right
   for ternary weights (updates happen at quantization boundaries). The
   matched experiment compares the inherited Muon split vs a plain-AdamW
   ternary baseline.
5. **Embeddings float first.** The notes recommend starting with float
   embeddings + output head and counting them honestly in deployed bytes.
   D1's "quantize everything" is kept as the max-compression variant; the
   *standard* variant quantizes only transformer matrices (and optionally
   value embeddings), keeping wte/lm_head in bf16. Both are measured.
6. **TII convention noted.** TII/onebitllms uses one scale *per tensor* and
   excludes lm_head; original BitNet b1.58 (and this project) uses
   per-output-channel scales. 2-bit packed uint8 storage is the common
   ground; llama.cpp TQ2_0/TQ1_0 supports the export path.
7. **Instrument the quantizer** (not just loss): per-layer +/-/0 fractions,
   trit flip rate per step, scale drift, export/reload agreement. Added to
   the ternary.py deliverables.
8. **Spectra caveat.** "Ternary models just need more tokens" is a
   simplification; allocation between params and tokens depends on the
   fitted curve. nanochat's auto size/LR/token choices are tuned for its
   fp recipe — treat them as a starting point for the ternary runs.

## D12. Autonomy architecture: OpenRouter-served pi + systemd standup (2026-11-10)

pi is currently served by OpenRouter (kimi-k3), NOT the local llama-server.
Consequences:
- The GPU is free for training WHILE pi runs — the swap.sh alternation is
  unnecessary in this configuration (it stays in the repo for a future
  local-served pi, and for Qwen-as-manager phases).
- Supervision loop: systemd --user timer `nanotern-standup.timer` fires
  `scripts/standup.sh` every 30 min (linger enabled, survives logout).
  standup.sh runs `pi --no-session -p <standup prompt>` — STATELESS on
  purpose: continuity lives in the repo (PLAN checkmarks, DECISIONS,
  INSIGHTS, STANDUP.md, logs/), not in a session file. flock prevents
  overlap. Rationale for --no-session: (a) observed 2026-11-10 that
  `pi --session <live session> -p ...` INJECTS the prompt into the live
  session — dangerous; (b) bounded token cost per wake-up.
- STANDUP.md is the standup agent's checklist: check training, fix or
  advance, record insights, never approve own claims, D-entry before big
  decisions.
- llama-server stays DOWN until the user says otherwise.
- Piece-1 result: bf16 matmul 8192³ = 163 TFLOPs (> 50 bar). Piece 5
  result: smoke 126M, 100 steps loss 10.34→7.57, val bpb 3.15→1.73;
  resume restored step 100 exactly and continued (bpb 1.57 at step 110).
  ~0.9M tok/s instantaneous at B=262144 tokens/step on the 4090.

## D11. Operating model & experiment governance (2026-11-09)

Agreed with the user:

1. **Autonomy.** pi (this agent) is in charge of the experiment and uses the
   swap.sh/handoff.sh machinery without user intervention. llama-server may
   be down for weeks; this is the user's personal workstation.
2. **Chain of command.** pi runs standup: kicks in every N training cycles to
   spot-check the work and the direction. Qwen 3.8 (27B, served by
   llama-server) is the *manager* of day-to-day test runs during agent
   phases; pi manages Qwen. Neither agent approves its own claims: every
   proposed improvement needs a hypothesized mechanism, a source-or-
   speculation label, a correctness test, a matched baseline, and an
   acceptance criterion (per USER_NOTES_TO_QWEN.md).
3. **Benchmark bar.** Qwen3.5-0.9B (dense fp) is the named peer for the 1B
   model's "same class" comparison, with the honest caveat that it trained
   on orders of magnitude more tokens; we also compare quality-per-byte
   (we are ternary; it is not) and against ternary-class peers (Spectra
   TriTera ~1B, BitNet b1.58) where numbers exist. Val bpb and small
   standard evals are the internal metrics.
4. **Post-training / "dirty train".** A later phase (new piece 12): lift the
   pretrained ternary model from "not very good" to "not too bad" via
   sequence-level distillation from license-permitting larger models
   (Qwen / DeepSeek / Kimi families — check the specific model license at
   the time; Qwen3 is Apache-2.0) + light preference optimization. We
   re-tokenize teacher generations with our own BPE, so the tokenizer-
   mismatch problem for logit-level KL (USER_NOTES) is sidestepped for the
   SFT stage; logit-level distillation (e.g. token-scaled KL, arXiv
   2308.06744) remains an option only where token alignments can be built.
5. **Insights log.** `INSIGHTS.md` at repo root. At the end of every agent
   session, before handoff to training, the agent answers: "Did anything
   occur to you while working through this that may be relevant to our
   efforts?" Answers are recorded there (dated, attributed to the agent).
   These pilot new experimental directions. handoff.sh appends a dated
   template entry so the step can't be silently skipped.

## D13. fp-control-126m results and PTQ baseline (2026-11-10)

Run `fp-control-126m`: 1,920 steps, 503,316,480 tokens, 1.13h on 4090.
Config `smoke` (8L/512d, 125.8M params), batch 262,144 (16×8×2048).
Final val bpb: **0.965944** (falling monotonically across evals).
Train loss: 3.1847 → 3.0104. Sample generation: repetitive / shallow
("The The The..."), expected at this token budget for a 126M model.

PTQ from the final checkpoint (`eval_bpb.py` on 2M-token beginning-of-val
window):

| variant | bpb | delta vs fp | size (disk) | compressed vs fp |
|---|---|---|---|---|
| bf16 fp | 0.944967 | — | 906 MB | — |
| ternary standard | 1.603953 | **+0.659** | 241 MB | 3.8× |
| ternary max-compression | 1.764400 | **+0.819** | 95 MB | 9.6× |

Generation from ternary-standard (greedy, "The quick brown fox"):
`, the popular type of fer and fer, and the fer will be re-centre...` —
clearly incoherent. Naive per-row PTQ on an undertrained 126M model is
catastrophic. This validates the D1 plan: PTQ-first to get the baseline
delta, then QAT/TTF as the experimental axes.

Action: matched QAT (piece 10) design doc is the next gated step; TTF
(piece 11) clone is prep work.

## D17. Official experiment sequencing (user-approved 2026-11-10)

1. Finish the 126M quantization science: QAT max-compression (running),
   seed-2 replicates of fp-control and qat-standard (size the noise floor),
   all-AdamW QAT variant (piece 10-ii), TTF matched run (piece 11; D-entry
   chooses port-into-train.py vs their trainer — leaning port, for
   tokenizer/data/eval comparability).
2. Piece 13 (curriculum, 4 arms incl. Anki closed-loop refresher) and
   piece 14 (good-views JEPA, factorial) on the fp control — BEFORE the 1B
   launch, because both study the pretraining recipe the 1B run will use.
   Corpus protocol first (frozen splits, balanced answers, shortcut
   controls, never-triggering eval set for the Anki arm).
3. tern1b throughput benchmark (20 steps, real config) → 1B launch D-entry
   (method = winner of step 1; recipe = winner of step 2).
4. The 1B run (~1 week exclusive GPU). CPU-only work during it: eval
   harness (hellaswag-style; bpb alone can't support the peer claim),
   distillation corpus planning.
5. Piece 12: ternarize → eval vs Qwen3.5-0.9B → SFT on teacher generations
   → light preference optimization. Framing: quality-per-byte is our win
   condition; general benchmarks vs Qwen3.5-0.9B are context, not the bar.

Risks (ranked): TTF fails to reproduce → fall back to QAT for the 1B;
1B underperforms → matched 126M science + method is the deliverable;
benchmark comparison looks bad → reframe quality-per-byte (always the real
claim).

## D10. GPU reality: llama-server cohabitation (measured 2026-11-09)

`llama-server` (Qwen3.8-27B-IQ4_NL, the model serving this very agent) holds
**23.6 GB** of the 24 GB 4090 — and the KV cache *grows with our own
conversation* (it caps at 32k context). Only ~1 GB is free right now.

Consequences:
- No GPU work is safely possible at all while the server is active and
  context is long — even a 37M model's CUDA context (~300-500 MB) races
  against the server's growing cache.
- Development proceeds on CPU (model smoke, tokenizer, tokenization) until
  the user decides to stop/restart the server (or rent a GPU).
- The big-run decision (PLAN piece 9) now has a concrete third option:
  **TTF native-ternary training of the 1.1B model on the 4090** (~3 GB
  optimizer state + trits; activations small with B=1-4 and short T; state
  offload to host RAM available if needed) — which is also the most
  on-theme way to "train a 1B 1.57-bit model from scratch on this machine".
- I will not kill the server myself (it would end this session).

## D14. Matched QAT baseline design (piece 10, 2026-11-10)

### Goal
Run the first conventional-QAT baseline on `smoke` (126M) to see if latent-weight STE training closes the +0.66 bpb gap vs PTQ, before testing TTF.

### Quantizer (same as PTQ, forward-only)
- Per-row threshold `t = mean(|row|)`, signs = {-1,0,+1}, scale `s = ||row||_2 / sqrt(nnz)`.
- 2-bit packed signs + fp16 scales at export; during training we keep unpacked `int8` signs in a temporary tensor and store only the dequantized float for backward.

### Architecture change
Replace eligible `nn.Linear` layers with `TernaryLinear` (subclass of `nn.Linear`).
Forward: `w_st = w + (dequantize(quantize(w)) - w).detach()` — straight-through estimator.
Backward flows to fp32/bf16 master `w` directly.

Candidate layers (standard variant):
- `transformer.h.*.{attn.{c_q,c_k,c_v,c_proj},mlp.{c_fc,c_proj}}`
- Exclude `ve_gate` (sensitive 12-channel → n_head gate; kept float per existing convention).
- `wte`, `lm_head`, `value_embeds` stay float.
Max-compression variant can add `value_embeds` and optionally `wte` later; not the first run.

### Optimizer
First run uses the **inherited Muon/AdamW split** (matched to fp-control-126m). `TernaryLinear.weight` is still a 2D matrix param, so it lands in Muon groups naturally.
Follow-up: all-AdamW variant (piece 10-ii) if Muon shows instability with STE.

### Training script changes
- Add `--qat` and `--qat-variant {standard,max-compression}` flags to `train.py`.
- Conversion happens after `init_weights()` and before `setup_optimizer()`.
- `load_checkpoint()` is compatible: `TernaryLinear` state dict keys are `weight`/`bias` (same as `Linear`).

### Instrumentation
- `val_bpb` is natively measured on the hardened ternary forward (no extra step).
- `trit_stats()` method on `TernaryLinear` reports per-layer fractions (-/0/+) and mean scale.
- Log trit stats every `--eval-every` step (piggyback on existing eval window).
- Track `roundtrip_rel_err` at init and after training to see if STE reduces quantization error.

### Check
Run `qat-126m-standard` for the **same 503M-token horizon** as fp-control-126m.
Accept if val bpb < PTQ standard (1.604) and ideally within ~0.1 of fp (0.966). Reject if it diverges or Muon shows pathological flip rates.

### Risks / mitigations
- **Divergence from STE saturation**: if too many weights get stuck near threshold, gradients vanish. Mitigate by monitoring zero-trit fraction; if it spikes > 0.9 on any layer, reduce LR or switch to AdamW.
- **Muon + ternary mismatch**: Muon orthogonalizes updates, but ternary boundary crossings are non-smooth. If loss curve is noisier than fp, try AdamW.
- **Resume compatibility**: checkpoint only stores master weights; QAT state is deterministic on load. No extra state needed.

## D15. Matched QAT baseline results (piece 10, 2026-11-10)

Run `qat-126m-standard`: 1,920 steps, 503,316,480 tokens, 1.19h on 4090.
Same config as `fp-control-126m` (`smoke`, B=262144), **from random init**
(same seed) with `--qat --qat-variant standard`.
CORRECTED 2026-11-10 morning: this entry originally claimed the run
"started from the fp-control checkpoint" — FALSE. Verified from
logs/qat-126m-standard.log: no "Resumed" line, step-0 loss 10.3973 and
val bpb 3.154745 identical to fp-control's fresh init, and standup #3's
INSIGHTS entry records step-10 loss 10.3405 ≈ fp-control's 10.3393.
So this IS the honest matched from-scratch ternary arm. (Meta-lesson:
the false sentence was written from memory, not from the log — exactly
the failure the verification standard exists to catch.)

Results:
- final val bpb: **1.025228** (vs fp 0.965944, PTQ standard 1.603953)
- bpb delta vs fp: **+0.059** (well within the ±0.10 acceptance bar)
- bpb delta vs PTQ: **−0.579** (QAT closes ~88% of the PTQ gap)
- train loss: 10.3973 → 3.2341 (monotonically falling)
- val bpb progression: 3.155 → 1.230 → 1.154 → 1.114 → 1.094 → 1.060 → 1.041 → 1.025 (falling at every eval)
- trit fractions stable across all layers at ~0.19 / 0.61 / 0.19 (-/0/+);
  no zero-trit spike > 0.9; no divergence.
- throughput: ~118k tok/s (vs fp ~123k tok/s; ~4% overhead from STE)
- Muon + STE showed no instability; loss curve is slightly noisier than fp
  post-step-1000 but continues to fall.

Acceptance: **PASS**. QAT is dramatically better than naive PTQ on this
undertrained 126M model and nearly matches the fp baseline. This validates
piece 10 and justifies TTF (piece 11) as the next experimental branch.

Next variants (deferred):
- `qat-126m-max-compression` (quantize value_embeds too)
- `qat-126m-all-adamw` (Muon vs AdamW ablation, per D14)

## D16. Process detachment: `setsid`, not `nohup`, inside systemd user services (2026-11-10)

`qat-126m-standard` was silently killed twice at ~step 40 when the standup
shell exited. `nohup` did not prevent SIGHUP inside the `nanotern-standup.service`
user unit, and `setsid` (new session + process group) also failed because
systemd's `KillMode=control-group` kills *all* processes in the service
cgroup on teardown, regardless of session detachment.

Actual fix: launch training via `systemd-run --user --scope --unit=...`
so it lives in a separate systemd scope unit outside the standup cgroup.
Defense-in-depth: add `KillMode=process` to `nanotern-standup.service`
so the service only kills its main process on stop, not any remaining
children (relevant for other launch paths).

Operational rule: any long-running training launched by standup uses
`systemd-run --user --scope --unit=<name> -- <command>` rather than
`nohup` or `setsid`.


## D18. QAT max-compression results and quality-per-byte comparison (2026-11-10)

Run `qat-126m-maxcomp`: 1,920 steps, 503,316,480 tokens, 1.21h on 4090.
Same config as `fp-control-126m` and `qat-126m-standard` (`smoke`, B=262144),
from random init with `--qat --qat-variant max-compression`.

### Verified deployed bpb (beginning-of-val, 2M tokens, eval_bpb.py)

| variant | val bpb | delta vs fp | deployed size (honest) | file size |
|---|---|---|---|---|
| fp control | 0.944967 | — | 251.7 MB | 865 MB |
| PTQ standard | 1.603953 | +0.659 | — | — |
| PTQ max-comp | 1.764400 | +0.819 | — | — |
| QAT standard | 0.997735 | +0.053 | 207.7 MB | 231 MB |
| QAT max-comp | 1.037754 | +0.093 | **61.2 MB** | **91 MB** |

QAT standard closes **92%** of the PTQ gap ((1.604−0.998)/(1.604−0.945)).
QAT max-comp closes **89%** of its PTQ gap.

### Quality-per-byte win

Max-compression costs only **+0.040 bpb** versus standard QAT but is
**3.4× smaller** deployed (61.2 MB vs 207.7 MB). Against the fp control,
it is **4.1× smaller** at a cost of +0.093 bpb. This makes max-comp the
preferred variant for the 1B run unless seed-2 replicates or the AdamW ablation
show instability.

### Cross-check: training-log val window

The training log's final val bpb (evaluated on the advanced val dataloader
position after 8 evals) was 1.024882 for max-comp and 1.025228 for standard.
On this sliding window the two variants are indistinguishable (+0.059 vs fp
for both). The beginning-of-val eval shows a small but consistent max-comp
penalty (+0.093 vs +0.053). Both windows agree that QAT nearly erases the
PTQ damage.

### Bug fix: eval_bpb.py on QAT checkpoints

`load_model_from_checkpoint` in `ternary.py` was loading QAT checkpoints into
plain `nn.Linear` layers, so `eval_bpb.py` evaluated the *master weights* in
full precision rather than the ternary quantized model. Master weights trained
with STE are not optimized for direct fp evaluation (bpb was 1.67 for max-comp,
2.27 for standard — worse than ternary). Fixed: when a checkpoint's meta
contains `args.qat=True`, `load_model_from_checkpoint` now calls
`convert_model_to_qat` before returning the model, restoring the true deployed
forward path.

### Decision

Accept both QAT variants. Proceed to seed-2 replicates (fp-control and
QAT-standard) to size the noise floor, then the all-AdamW ablation, then TTF.
