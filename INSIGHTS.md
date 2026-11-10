# Insights

Running log of answers to: **"Did anything occur to you while working through
this that may be relevant to our efforts?"** (D11)

Recorded at the end of every agent session, before handoff to training. Newest
first. These pilot new experimental directions — nothing here is a commitment,
but nothing here gets silently dropped either.

---

## 2026-11-10 — pi (standup #6: QAT finished)

- **`qat-126m-standard` PASS: val bpb 1.025** (+0.059 vs fp, −0.579 vs PTQ).
  STE+QAT closes ~88% of the naive-PTQ degradation on a 126M/503M-token run.
  This is the strongest evidence yet that latent-weight training recovers
  ternary quality. Next experimental axis is TTF (piece 11) — a matched
  126M run against this QAT baseline.
- **Throughput overhead of STE is ~4%** (118k vs 123k tok/s), negligible.
  The trit fraction ~61% zeros suggests the quantizer is using the ternary
  alphabet fully; no pathological collapse to all-zeros observed.

---

## 2026-11-10 — pi (standup #4: QAT restart, SIGHUP root-cause)

- **`nohup` is not enough inside systemd user services:** `qat-126m-standard`
  was silently killed twice at ~step 40 when the standup shell exited, even
  though previous standups believed `nohup` had detached it. `setsid` (new
  session + process group) is the minimal working fix: process now survives
  standup exit and is at step 60+ with loss dropping 10.34 → 8.66. Recorded
  in standup log and `DECISIONS.md` D15.

---

## 2026-11-10 — pi (standup #3: QAT run started)

- **Device-mismatch crash on QAT start:** `TernaryLinear` was defaulting to CPU
  device because `nn.Linear()` was called without `device=child.weight.device`.
  Fixed in `nanotern/qat.py` — both `__init__` and `convert_model_to_qat` now
  propagate device. Run `qat-126m-standard` is now live.
- **Step-10 loss is 10.3405, identical to fp-control (10.3393).** STE forward
  with random-init weights produces the same early-train loss as bf16, which
  suggests the model is not immediately killed by ternary quantization at init.
  The danger zone is mid-training divergence — watch for loss plateau or noisy
  spikes after step 500.

---

## 2026-11-10 — pi (standup #2: QAT design doc + core module)

- **QAT STE implementation is trivially local:** `TernaryLinear` subclassing
  `nn.Linear` with `w_st = w + (w_q - w).detach()` is ~20 lines and requires
  zero changes to optimizer or checkpoint logic. The 48 transformer matrices
  convert cleanly; `lm_head`, `ve_gate`, and `smear_gate` stay float per
  the existing standard-variant convention.
- **Training run is ready to start:** `train.py --qat --qat-variant standard`
  is wired. Next wake-up should either start the run (if GPU is free) or
  continue prep. llama-server is down (nvidia-smi showed no server process
  at 01:00), so the 4090 is available.

---

## 2026-11-10 — pi (standup: fp-control finished, PTQ measured)

- **PTQ delta is catastrophic: +0.66 bpb (standard) and +0.82 bpb (max-compression).**
  The ternary-standard model's greedy generation is complete gibberish
  (", the popular type of fer and fer, and the fer will be re-centre...").
  This is expected for naive per-row threshold ternary on a 126M model
  trained to only ~0.5B tokens — the weights are not stable enough for
  aggressive quantization. The important thing is we now have the *honest*
  baseline number; any QAT/TTF experiment that improves on this is real.
- **The quantizer's ~48% relative error is a diagnostic, not a root cause.**
  Per-layer rel_err is uniform (~0.48) which suggests the threshold
  quantizer is consistently losing signal; a learned threshold or STE
  during training (QAT) should push this down dramatically. Worth tracking
  rel_err alongside bpb in future ablations.
- **Sample generation from the fp model is also poor ("The The The..."),**
  confirming that 126M @ 0.5B tokens is the floor, not a fair test of
  ternary quality. Future PTQ/QAT comparisons should probably use a model
  trained to at least 2–3B tokens where the fp control is actually fluent.
  This may mean inserting a "train smoke to 2B tokens" run before piece
  10/11 comparisons. Piece 8 explicitly says the delta is the metric, but
  the absolute quality matters for whether the comparison is informative.

---

## 2026-11-10 — pi (first autonomous night: bring-up)

- **The dataloader bug is the vindication of the verification standard.**
  next_batch counted chunks not rows → B(B+1)/2 rows per batch. Every
  sanity check before it had passed; only the explicit piece-4 shape check
  caught it. At B=262144 tokens/step it would have silently inflated every
  batch — Power-Lines batch math, LR scaling, token horizons all poisoned
  invisibly. Lesson for Qwen-as-manager: the piece checks exist because
  code that "obviously works" doesn't.
- **126M smoke throughput: ~0.9M tok/s instantaneous** at B=262144 — far
  above the pessimistic D6 estimates. If tern1b scales even at 1/8 of that
  per-token rate (~110k... to be measured), the 6.5B-token run is ~16h,
  not days. Don't update D6 yet — measure the real config first.
- **`pi --session <live-session> -p` injects into the live session**
  (observed tonight). Standup uses --no-session; repo docs are the shared
  memory. If we ever want standup to have continuity, give it a DEDICATED
  session id, never the interactive one.
- Tokenizer trained in ~2 min on 8 threads (rustbpe); 465M train tokens
  tokenized in ~3 min. The CPU pipeline was never the bottleneck — GPU
  windows are the scarce resource, always.

## 2026-11-09 — pi (user's multi-modal/JEPA idea; verified arXiv 2610.39302)

- Balestriero & LeCun is real and strong: good-view JEPAs reach
  representation alignment 42x more sample-efficiently than NTP; a 109M
  text JEPA out-aligns 72B LMs vs vision encoders. For a project bounded
  by 6.5B tokens on one GPU, sample efficiency is the most valuable
  currency there is → PLAN piece 14.
- **All three of the user's ideas hang off one artifact**: the symbolic
  oracle generator feeds piece 13 (curriculum), the Anki arm (probe +
  refreshers), AND piece 14 (paired good views, oracle-verified to carry
  identical information). Build the generator well once; three experiment
  families unlock. Priority of generator work just went up.
- Caution carried from MULTIVIEW.md and worth repeating: the paper proves
  ALIGNMENT, not downstream quality. Alignment is our diagnostic; val bpb
  + probes remain the acceptance metric. Also NTP-as-bad-view-JEPA is a
  nice frame for why pure ClimbMix pretraining is sample-inefficient —
  documents are partial observations, not paired renderings.

## 2026-11-09 — pi (user's "Anki-style training" idea)

- User asked whether spaced-repetition-style refreshers have been tried.
  Yes in continual learning (replay buffers; small fractions — low single
  digit % — prevent most forgetting) and data-repetition spacing studies,
  and maximal-interfered-retrieval (NeurIPS 2019) is literally Anki's
  "review at the edge of forgetting" as a sampling rule. But closed-loop,
  probe-triggered refresher scheduling at LLM-pretraining scale appears
  untested → added as piece 13 arm (d). Cheap: probe is a few hundred
  held-out examples per val window, refresher spend ~1-2% of tokens.
- Design wrinkle worth remembering: the probe doubles as the retention
  metric, so the SAME probe can't also be the headline result — reserve a
  second, never-triggering symbolic eval set for final scoring, else arm
  (d) is graded on its own study cards.

## 2026-11-09 — pi (user's curriculum idea: symbolic-first pretraining)

- User proposal: borrow the neural-set-union *protocol* — symbolic corpora
  first, general text second, specialization last — as nanotern's
  pretraining curriculum. Recorded as PLAN piece 13 (three-arm ablation at
  126M, equal tokens, downstream-metric judged).
- **The tokenizer is the hidden coupling.** If any symbolic-stage data is
  in our future, a slice of it must be in the BPE training set — otherwise
  symbolic text tokenizes into byte-shrapnel and the ablation confounds
  curriculum with tokenization quality. Corpus generation must precede
  piece 3 (tok_train). This is the only time-ordering constraint the idea
  creates; everything else is free to reorder.
- **Synthetic data inverts the usual scarcity.** ClimbMix tokens cost
  download + tokenization; symbolic tokens cost nothing but the oracle's
  CPU time and are exactly labeled. At fixed GPU compute the question is
  purely "which token mix buys the most downstream quality" — a cleaner
  experimental question than most data ablations get.
- Literature note for future-us: frontier practice mixes-and-anneals
  rather than stages (OLMo 2, MiniCPM WSD, Llama anneal). Arm (b) of
  piece 13 exists because hard staging is the disfavored prior; if arm (a)
  wins anyway, that's a genuinely interesting result worth writing up.

## 2026-11-09 — pi (review of ~/git/neural-set-union, cross-project)

- **That project's failure mode is our failure mode.** Their pilot trained
  <1M tokens and got an ambiguous +1.23pp; the diagnosis was an
  *unestablished representation contract* — learned interfaces never fed the
  frozen operators exact operands. Direct analog for us: a ternary model
  trained briefly and judged underwhelming proves nothing about ternary.
  The matched fp control at equal tokens is what separates "ternary is
  hard" from "we undertrained / mis-wired."
- **Their governance ideas are worth stealing wholesale**: predeclared
  promotion thresholds ("a stage is not promoted because a handful of
  training prompts were memorized"), privileged/teacher modes marked and
  structurally excluded from scores, shortcut baselines (always-empty,
  source-copying, answer-frequency) reported alongside accuracy, frozen
  component checksums verified after training, and renaming/permutation
  equivariance tests. Our pieces 10/11 protocol should adopt these
  verbatim, especially: equivariance of val bpb under data-shard shuffles
  is our analog of their renaming tests.
- **"Frozen exact core + learned interfaces" is the same shape as
  "frozen quantizer + learned weights."** Their lesson: test the interface
  (grounding/routing) separately from the core. Ours: test the quantizer
  (instrumentation: trit fractions, flip rates, scale drift — already in
  piece 6) separately from the optimizer. A loss curve conflates them.
- **Their user-correction pattern maps to D11**: the manager agent drifted
  twice and needed a settled-constraints doc (RESEARCH_DIRECTION.md). Our
  DECISIONS.md plays that role for Qwen; when Qwen drifts, the fix is a
  D-entry, not a conversation.

## 2026-11-09 — pi (repo review + governance setup)

- **The 14–18 day big-run estimate was a units bug.** D6 wrote
  "6×1.1e9 FLOPs/token ≈ 6.6e15" — it's 6.6e9. Corrected throughput is
  ~10k tok/s, so the compute-optimal 1.1B run is ~6–9 days, and 500M–1B
  token budget runs are ~half a day to a day. Implication: the "budget
  run vs full run" tradeoff is much gentler than the docs suggested; we
  can afford a real intermediate-scale run (e.g. 126M for 2–3B tokens)
  as the matched-experiment vehicle instead of a tiny one.
- **Value embeddings are 46% of tern1b's parameters.** Ternary quantization
  of embedding *tables* is underexplored relative to Linear weights in the
  papers we surveyed — per-row thresholds on lookup tables may behave
  differently (rows are accessed discretely, never summed). The
  quantize-value-embeds-vs-not ablation (piece 10, variant iii) might be
  the most informative single ablation in the whole plan.
- **The backout/smear/resid-lambda machinery is untested under ternary.**
  These nanochat tricks exist to squeeze quality out of small fp models.
  Whether they survive sign-only weights is unknown; if the matched
  experiment underperforms, ablating backout (λ=0.2 subtraction!) is a
  prime suspect list item.
- **Checkpoint disk budget**: a tern1b fp32+optimizer checkpoint is ~13 GB
  and save_checkpoint keeps step_NNNNNNN.pt + latest.pt (~26 GB per save
  event). At 586 GB free this is ~20 saves before pressure. Worth a
  --max-checkpoints-retained flag before the big run, not during.

- **Background processes started in a bash tool session die on SIGHUP when the
  session ends.** The first `qat-126m-standard` run (PID 233909) disappeared
  after step 30 not because of a CUDA or code bug, but because the standup
  shell exited. Using `nohup` (or `systemd-run --user --scope`) is mandatory
  for any long-running training launched from an agent session. We should
  encode this in the launch wrapper script so future autonomous runs don't
  repeat the mistake.

## 2026-11-10 — pi (standup #5: QAT relaunch)

- **qat-126m-standard died again after step 90**, cause unknown (log stops cleanly, no traceback, GPU idle at next wake-up). Even with `setsid` + detached streams it did not reach step 250 / first checkpoint. Restarted with `PYTHONUNBUFFERED=1` for clearer live logs and `--save-every 250` to survive future early kills.
- **fp-control-126m is done** (val bpb 0.966, PTQ deltas recorded in D13). PTQ baseline established; QAT is the next experimental axis.


## 2026-11-10 — pi (standup #6: QAT scope fix)

- **Root cause of repeated QAT deaths was systemd cgroup cleanup**, not SIGHUP.
  `setsid` + detached streams still die because `nanotern-standup.service` has
  `KillMode=control-group` (default). Fixed by launching QAT via
  `systemd-run --user --scope --unit=nanotern-qat` and adding
  `KillMode=process` to the standup service.
- **QAT relaunched** in `nanotern-qat.scope`, currently at step 0, loss 10.3973.
  Should survive the standup shell exit and reach step 250 / first checkpoint
  before the next wake-up (~04:30).

---

## 2026-11-10 — pi (standup #7: QAT maxcomp done, D-entry written)

- **QAT maxcomp is the quality-per-byte winner.** 61.2 MB deployed vs 207.7 MB
  for standard QAT, at a cost of only ~+0.04 bpb (1.038 vs 0.998 on the same
  val window). That is a **3.4× size reduction** for a ~4% relative bpb hit.
  Unless seed-2 replicates show unacceptable variance, max-compression should be
  the default ternary variant for the 1B run.
- **QAT checkpoint evaluation trap discovered and fixed.** `eval_bpb.py` loads
  checkpoints via `load_model_from_checkpoint`, which creates a fresh `GPT` with
  `nn.Linear` layers. A QAT checkpoint only stores master weights, so evaluation
  was running the *full-precision* forward — and getting *worse* bpb (1.67 vs
  1.04) because master weights trained with STE are not optimized for fp
  inference. Fix: `load_model_from_checkpoint` now inspects `meta.args.qat` and
  re-applies `convert_model_to_qat` after loading, restoring the true deployed
  forward path. All future QAT evals are valid without manual export.
- **Added `--seed` flag to `train.py`** (default 42) so seed-2 replicates can
  proceed next.
