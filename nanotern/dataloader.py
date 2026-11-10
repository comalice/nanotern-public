"""
Streaming dataloader over pre-tokenized .bin shards (uint32, rows of T+1).

Each .bin row is [inputs (T) | targets (T)]. The loader mmaps the shards and
streams rows, cycling through the split infinitely. Resume state = (shard_idx,
row_offset) so training can restart exactly.

Usage:
    loader = make_dataloader("train", B=4, T=2048)
    for x, y in loader:
        ...
    loader.state_dict() / make_dataloader(..., resume_state=loader.state_dict())
"""
import os
import numpy as np
import torch

from nanotern.common import get_base_dir
from nanotern.tokenize import list_bin_files


class BinShard:
    """A single mmap'd .bin shard: N rows of T+1 uint32 tokens."""
    def __init__(self, path, T):
        self.path = path
        self.T = T
        self.row_bytes = (T + 1) * 4
        self.n_rows = os.path.getsize(path) // self.row_bytes
        self.map = np.memmap(path, dtype=np.uint32, mode='r')
        self.offset = 0  # row offset within this shard

    def rows(self, n, device="cuda"):
        """Return (x, y) batch tensors of n rows, wrapping within the shard."""
        T = self.T
        n_rows = self.n_rows
        idx = (self.offset + np.arange(n)) % n_rows
        # handle wrap with at most 2 copies
        raw = np.empty(n * (T + 1), dtype=np.uint32)
        base = self.offset
        first_chunk = min(n, n_rows - base) if base < n_rows else 0
        if first_chunk > 0:
            raw[:first_chunk * (T + 1)] = self.map[base * (T + 1):(base + first_chunk) * (T + 1)]
        rest = n - first_chunk
        if rest > 0:
            raw[first_chunk * (T + 1):] = self.map[:rest * (T + 1)]
        self.offset = (base + n) % n_rows
        arr = raw.reshape(n, T + 1)
        # nn.Embedding requires int32/int64 indices; .bin stores uint32
        x = torch.from_numpy(arr[:, :-1].astype(np.int64))
        y = torch.from_numpy(arr[:, 1:].astype(np.int64))
        if device != "cpu":
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        return x, y


class BinDataLoader:
    def __init__(self, split, B, T, device="cuda", resume_state=None):
        self.split = split
        self.B = B
        self.T = T
        self.device = device
        self.shards = [BinShard(p, T) for p in list_bin_files(split)]
        assert len(self.shards) > 0, f"No .bin shards for split={split}; run: python -m nanotern.tokenize --split {split}"
        self.shard_idx = 0
        if resume_state is not None:
            self.shard_idx = resume_state["shard_idx"]
            self.shards[self.shard_idx].offset = resume_state["row_offset"]
        self.step = resume_state["step"] if resume_state is not None else 0

    def next_batch(self):
        B = self.B
        x_rows, y_rows = [], []
        n_rows = 0
        while n_rows < B:
            need = B - n_rows
            avail = self.shards[self.shard_idx].n_rows - self.shards[self.shard_idx].offset
            take = min(need, avail) if avail > 0 else 0
            if take > 0:
                x, y = self.shards[self.shard_idx].rows(take, device="cpu")
                x_rows.append(x)
                y_rows.append(y)
                n_rows += take
            if take < need:  # wrap to next shard
                self.shard_idx = (self.shard_idx + 1) % len(self.shards)
                self.shards[self.shard_idx].offset = 0
        self.step += 1
        x = torch.cat(x_rows, 0)
        y = torch.cat(y_rows, 0)
        if self.device != "cpu":
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        return x, y

    def state_dict(self):
        return {
            "shard_idx": self.shard_idx,
            "row_offset": self.shards[self.shard_idx].offset,
            "step": self.step,
        }

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()


def make_dataloader(split, B, T, device="cuda", resume_state=None):
    return BinDataLoader(split, B, T, device=device, resume_state=resume_state)
