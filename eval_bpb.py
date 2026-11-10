"""Evaluate val bpb from a checkpoint (bf16 or ternary)."""
import argparse
import math
import torch
from nanotern.common import compute_init, print0
from nanotern.dataloader import make_dataloader
from nanotern.tokenizer import get_token_bytes
from nanotern.ternary import load_model_from_checkpoint

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none')
        loss2d = loss2d.view(-1)
        y = y.view(-1)
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-tokens", type=int, default=2_000_000)
    p.add_argument("--t", type=int, default=2048)
    args = p.parse_args()
    compute_init(seed=42)
    device = torch.device(args.device)
    model, ckpt = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    val_loader = make_dataloader("val", args.batch_size, args.t, device=str(device))
    token_bytes = get_token_bytes(device=str(device))
    eval_steps = args.eval_tokens // (args.t * args.batch_size)
    bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"val bpb: {bpb:.6f}")

if __name__ == "__main__":
    main()
