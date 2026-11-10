"""
Checkpoint save/load: model (fp32 master), optimizer state, and training meta.
Checkpoints live under <base_dir>/checkpoints/<run_name>/step_<N>.pt (+ latest.pt).
"""
import os
import glob
import torch

from nanotern.common import get_base_dir


def run_dir(run_name):
    d = os.path.join(get_base_dir(), "checkpoints", run_name)
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(run_name, step, model, optimizer, meta, tag=None):
    """Save a checkpoint. meta: dict of scalars/dataclass etc. Returns path."""
    d = run_dir(run_name)
    path = os.path.join(d, f"step_{step:07d}.pt")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "meta": meta,
    }
    torch.save(payload, path)
    # maintain a stable 'latest' pointer (copy, so a mid-write crash can't corrupt it)
    latest = os.path.join(d, "latest.pt")
    torch.save(payload, latest + ".tmp")
    os.replace(latest + ".tmp", latest)
    print(f"Saved checkpoint {path}")
    if tag is not None:
        tagged = os.path.join(d, f"{tag}.pt")
        torch.save(payload, tagged)
    return path


def find_latest(run_name):
    """Latest checkpoint path in a run dir, or None."""
    d = os.path.join(get_base_dir(), "checkpoints", run_name)
    steps = glob.glob(os.path.join(d, "step_*.pt"))
    if not steps:
        return None
    return sorted(steps)[-1]


def load_checkpoint(path, model, optimizer, device):
    """Load a checkpoint into model/optimizer. Returns (step, meta)."""
    print(f"Loading checkpoint {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    step = payload["step"]
    meta = payload.get("meta", {})
    return step, meta
