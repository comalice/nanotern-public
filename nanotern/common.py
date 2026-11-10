"""
Shared plumbing: base dir, compute dtype, seeding.
Mirrors the spirit of nanochat/common.py, single-GPU only.
"""
import os
import torch


def get_base_dir() -> str:
    """Root cache dir for data, tokenizer, checkpoints. Override with NANOTERN_BASE_DIR."""
    return os.environ.get("NANOTERN_BASE_DIR", os.path.expanduser("~/.cache/nanotern"))


def get_data_dir() -> str:
    d = os.path.join(get_base_dir(), "data")
    os.makedirs(d, exist_ok=True)
    return d


def autodetect_compute_dtype() -> torch.dtype:
    """bf16 on CUDA SM 8.0+ (like nanochat), fp32 elsewhere."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
    return torch.float32


COMPUTE_DTYPE = autodetect_compute_dtype()


def compute_init(device_type: str = "cuda", seed: int = 42) -> None:
    """Seed + TF32 settings. Call once at program start."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # TF32: allow for matmuls (nanochat uses "high")
    torch.set_float32_matmul_precision("high")


def print0(message: str) -> None:
    print(message, flush=True)
