"""
Training loop for nanotern (single-GPU).

Ports nanochat's hyperparameter system (scaling-law token horizon, Power-Lines
auto batch size, sqrt batch LR scaling, T_epoch weight-decay scaling, linear
warmup/constant/linear-warmdown LR, Muon momentum schedule, cosine WD decay)
to a single-GPU, pre-tokenized .bin data pipeline. See DECISIONS.md D2/D5/D6.

Usage:
  # GPU smoke: 200 steps of the 126M model
  python -m nanotern.train --config smoke --num-iterations 200

  # the big one (after user decision on budget)
  python -m nanotern.train --config tern1b --device-batch-size 8
"""
import os
import sys
import math
import time
import argparse
import dataclasses
import datetime

import torch

from nanotern.common import compute_init, print0, COMPUTE_DTYPE
from nanotern.model import make_config, GPT, CONFIG_PRESETS
from nanotern.dataloader import make_dataloader
from nanotern.checkpoint import save_checkpoint, load_checkpoint, find_latest, run_dir
from nanotern.tokenizer import get_token_bytes
from nanotern.qat import convert_model_to_qat, log_trit_stats


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Bits-per-byte (tokenization-independent): sum of nats over bytes, special
    tokens (byte count 0) excluded. Port of nanochat's loss_eval.evaluate_bpb.
    """
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none')  # (B, T)
        loss2d = loss2d.view(-1)
        y = y.view(-1)
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)


def build_meta_model(n_layer=12):
    """Reference d12 model (meta device, counting only) for the T_epoch math."""
    cfg = make_config("tern1b", n_layer=n_layer, n_head=6, n_kv_head=6, n_embd=768)
    with torch.device("meta"):
        return GPT(cfg)


def get_scaling_params(m):
    """transformer matrices + lm_head (nanochat's scaling-law param count)."""
    c = m.num_scaling_params()
    return c['transformer_matrices'] + c['lm_head']


def main():
    parser = argparse.ArgumentParser(description="Train a nanotern model")
    parser.add_argument("--config", choices=list(CONFIG_PRESETS.keys()), default="smoke")
    parser.add_argument("--run-name", type=str, default=None,
                        help="checkpoint dir name (default: <config>-<timestamp>)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--t", type=int, default=2048, help="sequence length (rows of the .bin data)")
    parser.add_argument("--device-batch-size", type=int, default=-1,
                        help="micro-batch size (-1 = preset default: tiny/smoke 8, tern1b 4)")
    parser.add_argument("--total-batch-size", type=int, default=-1,
                        help="total tokens per optimizer step (-1 = auto, Power-Lines)")
    parser.add_argument("--num-iterations", type=int, default=-1,
                        help="explicit optimizer step count (-1 = token horizon / batch)")
    parser.add_argument("--target-param-data-ratio", type=float, default=12.0)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=250, help="val bpb every N steps (-1 = off)")
    parser.add_argument("--eval-tokens", type=int, default=2_000_000)
    parser.add_argument("--sample-every", type=int, default=-1, help="generate samples every N steps (-1 = off)")
    parser.add_argument("--save-every", type=int, default=-1, help="checkpoint every N steps (-1 = end only)")
    parser.add_argument("--resume", type=str, default=None,
                        help="'auto' = latest checkpoint of run, or explicit .pt path")
    parser.add_argument("--qat", action="store_true",
                        help="enable QAT: ternary fake-quant forward (STE)")
    parser.add_argument("--qat-variant", choices=["standard", "max-compression"], default="standard",
                        help="which weights to ternarize during QAT")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for init and data shuffle")
    args = parser.parse_args()

    compute_init(seed=args.seed)
    device = torch.device(args.device)
    run_name = args.run_name or f"{args.config}-{datetime.datetime.now():%Y%m%d-%H%M%S}"

    # ---------------- model ----------------
    config = make_config(args.config, sequence_len=args.t)
    print0(f"Config {args.config}: {dataclasses.asdict(config)}")
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    if args.qat:
        convert_model_to_qat(model, args.qat_variant)
        print0(f"QAT enabled ({args.qat_variant}): eligible Linear layers use ternary forward with STE")

    param_counts = model.num_scaling_params()
    for k, v in param_counts.items():
        print0(f"{k:24s}: {v:,}")
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")
    print0(f"Compute dtype: {COMPUTE_DTYPE}")

    # ---------------- hyperparameters (nanochat scaling system) ----------------
    target_tokens = int(args.target_param_data_ratio * get_scaling_params(model))
    d12_ref = build_meta_model(12)
    D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
    B_REF = 2 ** 19  # 524288 tokens, optimal batch at d12 (empirical, from nanochat)

    total_batch_size = args.total_batch_size
    if total_batch_size == -1:
        predicted = B_REF * (target_tokens / D_REF) ** 0.383  # Power Lines: B ∝ D^0.383
        total_batch_size = 2 ** round(math.log2(predicted))
        print0(f"Auto batch size: {total_batch_size:,} tokens")

    batch_ratio = total_batch_size / B_REF
    batch_lr_scale = batch_ratio ** 0.5  # η ∝ √(B/B_ref)
    print0(f"Batch LR scale: {batch_lr_scale:.4f} (B={total_batch_size:,})")
    weight_decay_scaled = args.weight_decay * math.sqrt(batch_ratio) * (D_REF / target_tokens)
    print0(f"Weight decay: {args.weight_decay} -> {weight_decay_scaled:.6f} (T_epoch scaling)")

    preset_default_bsz = {"tiny": 8, "smoke": 8, "tern1b": 4}[args.config]
    device_batch_size = args.device_batch_size if args.device_batch_size > 0 else preset_default_bsz
    tokens_per_step = device_batch_size * args.t
    grad_accum = max(1, total_batch_size // tokens_per_step)
    if grad_accum * tokens_per_step != total_batch_size:
        print0(f"WARNING: total batch not divisible by micro-batch; "
               f"using {grad_accum * tokens_per_step:,} tokens/step")
        total_batch_size = grad_accum * tokens_per_step

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    else:
        num_iterations = max(1, target_tokens // total_batch_size)
    total_tokens = num_iterations * total_batch_size
    print0(f"Training horizon: {num_iterations:,} steps = {total_tokens:,} tokens "
           f"(target: {target_tokens:,} @ ratio {args.target_param_data_ratio})")

    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        matrix_lr=args.matrix_lr * batch_lr_scale,
        scalar_lr=args.scalar_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
    )

    # ---------------- schedule ----------------
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters

    def get_lr_multiplier(it):
        if it < args.warmup_steps:
            return (it + 1) / args.warmup_steps
        elif it <= warmdown_start:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * args.final_lr_frac

    def get_muon_momentum(it):
        if it < 400:
            frac = it / 400
            return (1 - frac) * 0.85 + frac * 0.97
        elif it >= warmdown_start:
            progress = (it - warmdown_start) / warmdown_iters
            return 0.97 * (1 - progress) + 0.90 * progress
        else:
            return 0.97

    def get_weight_decay(it):
        return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / max(1, num_iterations - 1)))

    # ---------------- data ----------------
    train_loader = make_dataloader("train", device_batch_size, args.t, device=str(device))
    val_loader = make_dataloader("val", max(1, device_batch_size), args.t, device=str(device)) \
        if args.eval_every > 0 else None
    token_bytes = None
    if val_loader is not None:
        token_bytes = get_token_bytes(device=str(device))
    eval_steps = (args.eval_tokens // (args.t * max(1, device_batch_size))) if val_loader else 0

    # ---------------- resume ----------------
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0.0
    if args.resume:
        path = args.resume
        if path == "auto":
            path = find_latest(run_name)
            assert path is not None, f"No checkpoints found for run {run_name}"
        step, meta = load_checkpoint(path, model, optimizer, device)
        val_bpb = meta.get("val_bpb")
        min_val_bpb = meta.get("min_val_bpb", float("inf"))
        smooth_train_loss = meta.get("smooth_train_loss", 0.0)
        ds = meta.get("data_state")
        if ds:
            train_loader.shard_idx = ds["shard_idx"]
            train_loader.shards[train_loader.shard_idx].offset = ds["row_offset"]
        print0(f"Resumed from step {step} (val_bpb={val_bpb})")

    log_path = os.path.join(run_dir(run_name), "train.log")
    def logline(msg):
        print0(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # ---------------- training loop ----------------
    model.train()
    t_start = time.time()
    tokens_this_proc = 0  # windowed throughput (resume-safe; lifetime tokens = step * B)
    logline(f"=== run {run_name}: step {step} -> {num_iterations}, "
            f"B={total_batch_size}, micro={device_batch_size}x{args.t}, accum={grad_accum} ===")

    for it in range(step, num_iterations):
        # schedule
        lr_mult = get_lr_multiplier(it)
        momentum = get_muon_momentum(it)
        wd = get_weight_decay(it)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_mult
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = wd

        # gradient accumulation
        model.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = next(train_loader)
            loss = model(x, y)
            loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()

        optimizer.step()

        smooth_train_loss = loss_accum if it == 0 else 0.99 * smooth_train_loss + 0.01 * loss_accum
        tokens_this_proc += total_batch_size
        elapsed = time.time() - t_start
        tok_s = tokens_this_proc / max(elapsed, 1e-9)  # this process's average throughput
        remaining = (num_iterations - it - 1) * total_batch_size / max(tok_s, 1e-9) / 3600

        if it % 10 == 0 or it == num_iterations - 1:
            logline(f"step {it:6d} | loss {smooth_train_loss:.4f} | tok/s {tok_s:,.0f} | "
                    f"lr {optimizer.param_groups[-1]['lr']:.5f} | "
                    f"~{remaining:.1f}h left")

        # evaluation
        if args.eval_every > 0 and it % args.eval_every == 0:
            model.eval()
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            min_val_bpb = min(min_val_bpb, val_bpb)
            model.train()
            logline(f"step {it:6d} | val bpb {val_bpb:.6f} (min {min_val_bpb:.6f})")
            if args.qat:
                for line in log_trit_stats(model, args.qat_variant):
                    logline(f"step {it:6d} | trits {line}")

        # samples (skip step 0: untrained samples are noise anyway)
        if args.sample_every > 0 and it > 0 and it % args.sample_every == 0:
            model.eval()
            sample_dir = os.path.join(run_dir(run_name), "samples")
            os.makedirs(sample_dir, exist_ok=True)
            with torch.inference_mode():
                toks = list(model.generate([0], max_tokens=200, temperature=0.8, top_k=100, seed=it))
            try:
                from nanotern.tokenizer import get_tokenizer
                text = get_tokenizer().decode(toks)
            except Exception:
                text = f"<decode unavailable> ids={toks[:20]}..."
            with open(os.path.join(sample_dir, f"step_{it:07d}.txt"), "w") as f:
                f.write(text)
            model.train()

        # checkpointing (skip step 0: an untrained tern1b checkpoint is ~13 GB of nothing)
        if it > 0 and ((args.save_every > 0 and it % args.save_every == 0) or it == num_iterations - 1):
            save_checkpoint(run_name, it + 1, model, optimizer, {
                "config": dataclasses.asdict(config),
                "args": vars(args),
                "val_bpb": val_bpb,
                "min_val_bpb": min_val_bpb,
                "smooth_train_loss": smooth_train_loss,
                "data_state": train_loader.state_dict(),
                "run_name": run_name,
            })

    # final save only for single-step runs (the loop already saves the last step)
    if num_iterations == 1:
        save_checkpoint(run_name, num_iterations, model, optimizer, {
            "config": dataclasses.asdict(config),
            "args": vars(args),
            "val_bpb": val_bpb,
            "min_val_bpb": min_val_bpb,
            "smooth_train_loss": smooth_train_loss,
            "data_state": train_loader.state_dict(),
            "run_name": run_name,
        })
    total_time_h = (time.time() - t_start) / 3600
    logline(f"=== done: {num_iterations:,} steps, {total_tokens:,} tokens in {total_time_h:.2f}h, "
            f"final val bpb {val_bpb} ===")


if __name__ == "__main__":
    main()
