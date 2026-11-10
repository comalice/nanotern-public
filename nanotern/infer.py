"""
Autoregressive generation from a bf16 or ternary checkpoint (PLAN piece 7).

Ternary checkpoints are dequantized to the compute dtype at load (D7): the
forward path is identical to bf16. Naive decode (no KV cache) — fine at our
scales; a fused ternary GEMM / KV cache are later optimizations, not
correctness requirements.

Usage:
  python -m nanotern.infer --checkpoint <ckpt.pt> --prompt "Hello" \
      --max-tokens 100 --temperature 0.8 --top-k 100
  python -m nanotern.infer --checkpoint <ckpt.pt> --prompt "2+2=" --temperature 0
"""
import argparse

import torch

from nanotern.ternary import load_model_from_checkpoint


def main():
    p = argparse.ArgumentParser(description="Generate text from a nanotern checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    fmt = ckpt.get("format", "bf16")
    print(f"loaded {args.checkpoint} (format={fmt}, step={ckpt.get('step')}, "
          f"variant={ckpt.get('variant', '-')})")

    from nanotern.tokenizer import get_tokenizer
    tok = get_tokenizer()
    tokens = tok.encode(args.prompt, prepend=tok.get_bos_token_id())
    for token in model.generate(tokens, max_tokens=args.max_tokens,
                                temperature=args.temperature, top_k=args.top_k,
                                seed=args.seed):
        print(tok.decode([token]), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
