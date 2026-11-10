"""
Ternary (1.57-bit) quantization: signs {-1,0,+1} + per-row fp16 scales,
2-bit packed sign storage. See DECISIONS.md D1 (quantizer) and D9 (variants).

Quantizer (per row of each 2D matrix):
  threshold t = mean(|row|);  sign = +1 if w > t, -1 if w < -t, else 0
  scale s = ||row||_2 / sqrt(nnz(row));  dequant: W ~= s * sign

Variants (D9, both measured):
  standard:        transformer Linear matrices only (wte/lm_head/value_embeds
                   stay in the compute dtype) — honest deployed-byte count
  max-compression: also quantize all embedding tables (wte, value_embeds;
                   lm_head stays float — it is both huge and sensitive)

Instrumentation (D9): per-layer trit fractions (+/-/0), scale stats,
roundtrip relative error. Trit flip rate per step is a QAT-time metric and
lives with the QAT code (piece 10), not here.

CLI:
  python -m nanotern.ternary --from-checkpoint <ckpt.pt> --out <tckpt.pt> \
      --variant standard
"""
import argparse
import os

import torch

from nanotern.common import COMPUTE_DTYPE, get_base_dir

TRIT_TO_BITS = {0: 0b00, 1: 0b01, -1: 0b10}  # 0b11 unused
BITS_TO_TRIT = torch.tensor([0, 1, -1, 0], dtype=torch.int8)


def quantize_rows(w):
    """w: (rows, cols) float tensor -> (signs int8 (rows, cols) in {-1,0,1},
    scales fp16 (rows,))."""
    assert w.ndim == 2
    w = w.float()
    t = w.abs().mean(dim=1, keepdim=True)                       # (rows, 1)
    signs = torch.zeros_like(w, dtype=torch.int8)
    signs[w > t] = 1
    signs[w < -t] = -1
    nnz = (signs != 0).sum(dim=1).clamp_min(1)
    scales = (w.norm(dim=1) / nnz.sqrt()).clamp_min(1e-8)       # (rows,)
    return signs, scales.half()


def dequantize_rows(signs, scales):
    return signs.float() * scales.float().unsqueeze(1)


def pack_signs(signs):
    """(rows, cols) int8 in {-1,0,1} -> uint8 (rows, ceil(cols/4)), 2 bits/trit."""
    rows, cols = signs.shape
    pad = (-cols) % 4
    if pad:
        signs = torch.cat([signs, torch.zeros(rows, pad, dtype=torch.int8)], dim=1)
    s = signs.view(rows, -1, 4).long()
    codes = torch.zeros_like(s)
    codes[s == 1] = 0b01
    codes[s == -1] = 0b10
    packed = (codes[..., 0] << 6) | (codes[..., 1] << 4) | (codes[..., 2] << 2) | codes[..., 3]
    return packed.to(torch.uint8), cols


def unpack_signs(packed, cols):
    """Inverse of pack_signs. -> (rows, cols) int8."""
    rows = packed.shape[0]
    p = packed.long().view(rows, -1, 1)
    codes = torch.cat([(p >> 6) & 3, (p >> 4) & 3, (p >> 2) & 3, p & 3], dim=2)
    signs = BITS_TO_TRIT.to(packed.device)[codes].view(rows, -1)
    return signs[:, :cols].contiguous()


# ---------------------------------------------------------------------------
# Checkpoint-level quantization
# ---------------------------------------------------------------------------

def classify_param(name, variant):
    """Should this parameter be ternarized? Only 2D weights are candidates."""
    if variant == "standard":
        # transformer block matrices only (attention + MLP Linears)
        return name.startswith("transformer.h.") and name.endswith(".weight") and "gate" not in name
    if variant == "max-compression":
        if name.endswith(".weight") and (
            name.startswith("transformer.h.") or name.startswith("transformer.wte")
            or name.startswith("value_embeds.")) and "gate" not in name:
            return True
        return False
    raise ValueError(variant)


def quantize_state_dict(state_dict, variant="standard"):
    """-> (ternary payload dict, report list[str]). 2D candidate weights become
    packed ternary entries; everything else is carried through as-is."""
    payload, report = {}, []
    total_params = quantized_params = 0
    for name, tensor in state_dict.items():
        if tensor.ndim == 2 and classify_param(name, variant):
            signs, scales = quantize_rows(tensor)
            packed, orig_cols = pack_signs(signs)
            payload[name] = {
                "ternary": True, "packed": packed, "cols": orig_cols,
                "scales": scales,
            }
            # instrumentation
            w = tensor.float()
            deq = dequantize_rows(signs, scales)
            rel_err = (deq - w).norm() / w.norm().clamp_min(1e-12)
            frac = [(signs == s).float().mean().item() for s in (-1, 0, 1)]
            report.append(
                f"  {name:44s} -/0/+: {frac[0]:.3f}/{frac[1]:.3f}/{frac[2]:.3f} "
                f"rel_err {rel_err:.4f} scale mean {scales.float().mean():.4f}")
            quantized_params += tensor.numel()
        else:
            payload[name] = {"ternary": False, "tensor": tensor}
        total_params += tensor.numel()
    report.append(f"quantized {quantized_params/1e6:.1f}M / {total_params/1e6:.1f}M params "
                  f"({100*quantized_params/max(total_params,1):.1f}%)")
    return payload, report


def dequantize_state_dict(payload):
    """Inverse of quantize_state_dict: full fp32 state_dict."""
    state_dict = {}
    for name, entry in payload.items():
        if entry["ternary"]:
            signs = unpack_signs(entry["packed"], entry["cols"])
            state_dict[name] = dequantize_rows(signs, entry["scales"])
        else:
            state_dict[name] = entry["tensor"]
    return state_dict


def payload_nbytes(payload):
    """Honest deployed-byte count: packed signs + fp16 scales + float tensors."""
    total = 0
    for entry in payload.values():
        if entry["ternary"]:
            total += entry["packed"].nelement() + entry["scales"].nelement() * 2
        else:
            # non-quantized tensors are deployed in the compute dtype
            total += entry["tensor"].nelement() * torch.tensor([], dtype=COMPUTE_DTYPE).element_size()
    return total


def quantize_checkpoint(ckpt, variant="standard"):
    """ckpt: loaded train.py checkpoint payload. Returns ternary checkpoint."""
    # keep stored dtypes: embeddings are COMPUTE_DTYPE (bf16 on GPU runs),
    # Linear master weights fp32; quantize_rows casts to float internally.
    # Deployed-byte accounting uses each tensor's stored element size.
    state_dict = dict(ckpt["model"])
    payload, report = quantize_state_dict(state_dict, variant)
    nbytes = payload_nbytes(payload)
    report.append(f"deployed size: {nbytes/1e6:.1f} MB "
                  f"({nbytes*8/state_dict_numel(state_dict):.2f} bits/param overall)")
    return {"format": "ternary", "variant": variant, "model": payload,
            "step": ckpt.get("step"), "meta": ckpt.get("meta", {})}, report


def state_dict_numel(state_dict):
    return sum(v.numel() for v in state_dict.values())


def load_model_from_checkpoint(path, device="cpu"):
    """Load either a train.py checkpoint or a ternary checkpoint into a GPT."""
    from nanotern.model import GPT, GPTConfig
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["meta"]["config"])
    model = GPT(cfg)
    model.init_weights()  # allocates real tensors; values overwritten below
    if ckpt.get("format") == "ternary":
        state_dict = dequantize_state_dict(ckpt["model"])
    else:
        state_dict = ckpt["model"]
    model.load_state_dict(state_dict)
    # If this was a QAT training checkpoint, restore TernaryLinear layers so
    # evaluation uses the same quantized forward path as training.
    args = ckpt.get("meta", {}).get("args", {})
    if args.get("qat") and args.get("qat_variant"):
        from nanotern.qat import convert_model_to_qat
        convert_model_to_qat(model, args["qat_variant"])
    return model.to(device), ckpt


def main():
    p = argparse.ArgumentParser(description="Quantize a checkpoint to ternary (2-bit packed)")
    p.add_argument("--from-checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--variant", choices=["standard", "max-compression"], default="standard")
    args = p.parse_args()

    ckpt = torch.load(args.from_checkpoint, map_location="cpu", weights_only=False)
    tckpt, report = quantize_checkpoint(ckpt, args.variant)
    print(f"variant={args.variant}")
    for line in report:
        print(line)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(tckpt, args.out)
    actual = os.path.getsize(args.out)
    print(f"wrote {args.out} ({actual/1e6:.1f} MB on disk)")


if __name__ == "__main__":
    main()
