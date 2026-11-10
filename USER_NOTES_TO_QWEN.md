Hello, Qwen, here are some notes on the latest research in training that may be relevant to your efforts here:

**Not foolhardy as a research project. Potentially foolhardy if the goal is “a broadly competitive 1B assistant, pretrained from scratch, on one 4090.”** Those are different targets.

A 4090 is a reasonable platform for developing a ternary training recipe, running controlled experiments, and producing a useful specialized model. The trap is assuming that **ternary inference efficiency translates into equally cheap training**. With conventional quantization-aware training, it generally doesn’t.

I checked recent research and nanochat’s current documentation. Here’s what I’d prioritize for your situation.

### 1. Master-weight-free training: the most directly relevant new direction

[**Training Ternary Language Models Without Master Weights — October 2026**](https://guzeldereli.dev/papers/ttf.pdf) is unusually relevant to your hardware constraint.

Conventional BitNet-style training maintains floating-point latent weights, quantizes them for the forward pass, and updates the latent weights through a straight-through estimator. You still pay for substantial floating-point training state.

This work proposes **two-timescale threshold flips, or TTF**:

- Keep the actual weights packed as trits.
- Accumulate normalized gradient evidence in a low-rank state.
- Change a trit when accumulated evidence crosses a threshold.
- Use a factored second moment rather than a per-weight Adam second moment.

The author reports comparisons from 110M through 2.9B parameters, with results approaching or matching a master-weight baseline at sufficient state rank. The [implementation and experiment descriptions](https://github.com/oguzeldereli/ttf-ternary) are available.

**Why it matters:** this attacks *training memory*, rather than merely making the final checkpoint smaller.

**Caution:** it is very recent author-reported work, not something I’d treat as an independently established replacement for ordinary QAT. Some configurations also include floating-point adapters and scales, so compare like with like.

**My recommendation:** reproduce a small matched experiment before porting it into your main 1B run. It deserves an experimental branch—not an immediate rewrite.

### 2. Soft-to-hard ternarization: don’t assume hard STE is the best optimizer interface

[**Hestia — January 2026**](https://arxiv.org/abs/2601.20745) replaces hard rounding during early QAT with a temperature-controlled soft relaxation, then progressively hardens the assignments. It uses a Hessian-derived sensitivity signal to adjust that schedule across tensors.

Its reported experiments include **1B and 3B models**, making it more relevant than demonstrations confined to tiny language models.

The actionable idea is:

> Let sensitive tensors adapt before forcing them onto a rigid ternary grid.

However, its evidence concerns QAT of pretrained models; **that does not establish that the same recipe wins when training your nanochat model from random initialization**.

For your project, I would compare:

- Hard ternary forward pass with a conventional STE.
- A simple soft-to-hard schedule.
- Only then, sensitivity-aware schedules.

Always evaluate the **fully hardened model**. A soft training checkpoint can look good while hiding an eventual discretization cliff.

### 3. Spectra/TriTera: useful scaling evidence, but read beyond the abstract

Two papers are particularly useful:

- [**Surprising Effectiveness of Pretraining Ternary Language Models at Scale — ICLR 2025**](https://proceedings.iclr.cc/paper_files/paper/2025/file/6499f26637f74f7c0fbfb46668434973-Paper-Conference.pdf)
- [**Scaling Laws and Efficient Inference for Ternary Language Models — ACL 2025**](https://aclanthology.org/2025.acl-long.1294/)

They provide controlled comparisons and ternary models near your intended scale.

The important distinction is **equal parameter count versus equal deployed size versus equal training compute**. Ternary models can look excellent per stored bit without matching an equal-parameter floating-point model in every metric.

Also, beware the simplified takeaway “ternary models just need more tokens.” The [author’s detailed scaling discussion](https://ayushk4.github.io/projects/spectra-1-1/) explains that the preferred allocation between parameters and tokens depends on where you are on the fitted curve. It does **not** justify unlimited overtraining as your best use of one GPU.

**For nanochat:** its automatic size, learning-rate, and token-budget choices were tuned for its existing architecture and training recipe. Once you change the weight representation and optimizer dynamics, treat those defaults as a starting point—not a validated ternary scaling law.

### 4. Recent PTQ changes the alternative you should benchmark against

[**CAT-Q — June 2026**](https://arxiv.org/html/2606.26650v1) reports ternarization of existing pretrained models using learnable modulation and softened ternarization with a small calibration set.

This is not a drop-in recipe for your from-scratch model. Its importance is strategic:

- If your objective is **understanding native ternary learning**, keep pursuing from-scratch training.
- If your objective is **the best useful ternary model you can obtain with your hardware**, conversion plus recovery training deserves a baseline.

One crucial caveat: “uses far fewer quantization tokens” does **not** mean “uses far less total training compute.” The converted model inherits the expensive pretraining of its source model.

I would avoid comparing a lightly pretrained scratch model against a converted, extensively pretrained model and concluding that the quantizer alone explains the difference.

### 5. Existing ternary training tooling gives you a valuable reference implementation

The [**Axolotl/TII ternary training integration — April 2026**](https://huggingface.co/blog/axolotl-ai-co/finetuning-ternary-llms-tii-axolotl) documents practical training and fine-tuning support, including a Falcon-E 1B latent-weight checkpoint.

Even if you stay with nanochat, this is useful for checking:

- Weight and activation quantization behavior.
- Which layers remain floating point.
- Export and packing behavior.
- Whether training-time and deployed computations agree.

**Don’t make your first experiment simultaneously test a new architecture, quantizer, optimizer, and export format.** A reference implementation lets you isolate which component is responsible when something breaks.

### 6. Packing and kernels matter—but they are a separate research problem

[**Breaking the 1.58-bit Barrier for Ternary LLMs — September 2026**](https://arxiv.org/abs/2609.16338) introduces BITCOS, exploiting nonuniform ternary symbol distributions for more compact storage.

Useful deployment research, but its reported GPU results concern Intel Xe2 hardware—not your 4090.

Similarly, Spectra’s TriRun kernels demonstrate that packed ternary weights can translate into real GPU inference gains under tested conditions. That is **not evidence that your fake-quantized PyTorch training loop will accelerate**.

For you, keep three measurements separate:

1. Training throughput and peak memory.
2. Actual packed checkpoint size, including floating-point tensors and metadata.
3. End-to-end inference speed on the intended deployment hardware.

“Only additions are needed mathematically” is not a hardware benchmark.

### What I would do on your 4090

These are my engineering recommendations, rather than claims that one paper establishes an optimal recipe.

**First, establish a smaller matched baseline.** Train a floating-point control and a conventional ternary-QAT version with identical data, tokenizer, architecture, and evaluation windows. Then test optimizer or quantizer changes individually.

**Specifically audit nanochat’s optimizer split.** [Nanochat uses Muon and AdamW](https://github.com/karpathy/nanochat/). Do not assume that an optimizer tuned for continuous matrix updates is automatically optimal for latent weights whose useful changes occur at quantization boundaries. Compare the inherited split with an AdamW-based ternary baseline before attributing a failure to ternary capacity.

**Start with floating-point embeddings and output head.** Count them honestly in both parameter totals and deployed bytes. They can be a substantial fraction of a small model, but quantizing everything immediately introduces another failure mode.

**Instrument the quantizer, not just the loss:**

- Fraction of negative, zero, and positive weights per layer.
- Fraction of weights changing ternary state per optimizer step.
- Scale drift and activation outliers.
- Gradient and update norms.
- Hard-quantized validation loss.
- Export/reload agreement with the training-time hard model.

**Budget from measured throughput.** Benchmark the actual 1B configuration—including activation checkpointing, quantization overhead, optimizer updates, and gradient accumulation—before committing to a long token budget. Neither packed inference size nor another GPU’s speedrun predicts your training time reliably.

### About Qwen 3.8 27B supervising the effort

“Supervising” could mean two different things, and I wouldn’t conflate them.

**If you mean coding/research supervision:** use it to propose patches, write tests, inspect profiler traces, and suggest ablations. Don’t let it approve its own claims. Require every proposed improvement to specify:

- The hypothesized mechanism.
- A source or explicit statement that it is speculative.
- A correctness test.
- A matched baseline.
- A measurable acceptance criterion.

A training loss that declines does not prove the STE, gradient accumulation, masking, or exported model is correct.

**If you mean teacher-model supervision:** offline generated examples are the simplest starting point. Logit distillation is another option, but naïve tokenwise KL requires aligned token semantics and prediction positions—not merely matching vocabulary sizes. A custom nanochat tokenizer and Qwen tokenizer are not automatically compatible.

An older but directly relevant reference is [**Token-Scaled Logit Distillation for Ternary Weight Generative Language Models**](http://arxiv.org/abs/2308.06744v4). It studies logit supervision specifically for ternary QAT, though its setting does not directly establish the best approach for a randomly initialized 1B student supervised by a different 27B architecture.

**Bottom line:** I’d prioritize a trustworthy conventional-QAT baseline, then test **TTF for training-memory savings** and **soft-to-hard ternarization for optimization quality**. Your most valuable result may be a clean, reproducible recipe with honest quality-per-byte and quality-per-GPU-hour measurements—not simply reaching 1B parameters.

