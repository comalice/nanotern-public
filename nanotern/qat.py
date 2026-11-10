"""
Quantization-Aware Training (QAT): TernaryLinear with straight-through estimator.

Forward uses the same per-row ternary quantizer as PTQ (ternary.py).
Backward flows unmodified to the fp32/bf16 master weight.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanotern.ternary import quantize_rows, dequantize_rows, classify_param


class TernaryLinear(nn.Linear):
    """
    nn.Linear that quantizes its weight to {-1, 0, +1} per row during forward.
    Gradients reach the underlying master weight via STE:
        w_st = w + (w_q - w).detach()
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None):
        super().__init__(in_features, out_features, bias=bias, device=device)

    def forward(self, x):
        w = self.weight
        signs, scales = quantize_rows(w)               # non-differentiable
        w_q = dequantize_rows(signs, scales)          # float reconstruction
        w_st = w + (w_q - w).detach()                 # STE
        return F.linear(x, w_st.to(x.dtype), self.bias)

    @torch.no_grad()
    def trit_stats(self):
        """Return (frac_neg, frac_zero, frac_pos, mean_scale)."""
        signs, scales = quantize_rows(self.weight)
        frac = [(signs == s).float().mean().item() for s in (-1, 0, 1)]
        return frac + [scales.float().mean().item()]


def convert_model_to_qat(model, variant="standard"):
    """
    Recursively replace eligible Linear layers in a GPT with TernaryLinear.
    Embedding tables are left untouched (honest standard variant).
    """
    def _replace(module, name):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            param_name = f"{full_name}.weight"
            if isinstance(child, nn.Linear) and classify_param(param_name, variant):
                tlin = TernaryLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                )
                if child.weight is not None:
                    tlin.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    tlin.bias.data.copy_(child.bias.data)
                setattr(module, child_name, tlin)
            else:
                _replace(child, full_name)

    _replace(model, "")
    return model


def log_trit_stats(model, variant="standard"):
    """Collect per-layer trit stats from TernaryLinear layers."""
    lines = []
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            stats = module.trit_stats()
            lines.append(
                f"  {name:50s} -/0/+: {stats[0]:.3f}/{stats[1]:.3f}/{stats[2]:.3f} "
                f"scale_mean {stats[3]:.4f}"
            )
    return lines
