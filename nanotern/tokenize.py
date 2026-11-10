"""
Pre-tokenize parquet shards into packed .bin files.

Each output row has exactly T+1 uint32 tokens: [inputs (T) | targets (T)].
Packing is nanochat's BOS-aligned best-fit:
  - every row starts with BOS
  - from a buffer of docs, repeatedly place the LARGEST doc that fits
  - when no doc fits, crop the SHORTEST buffered doc to fill the row exactly
  - 100% utilization, no padding; ~35% of tokens are cropped away at T=2048

Usage:
  python -m nanotern.tokenize --split train --n-shards 10
  python -m nanotern.tokenize --split val
"""
import os
import argparse
import numpy as np

from nanotern.common import get_base_dir
from nanotern.dataset import parquets_iter, list_parquet_files
from nanotern.tokenizer import get_tokenizer


def bestfit_pack(text_iter, tokenizer, T, buffer_size=1000):
    """
    Pack a stream of documents (from parquets_iter) into rows of T+1 uint32
    tokens using BOS-aligned best-fit packing. Yields np.uint32 arrays.
    Stops when the document stream is exhausted (no padding of final rows).
    """
    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []

    def refill():
        texts, _ = next(text_iter)
        ids_list = tokenizer.encode(texts, prepend=bos_token, num_threads=8)
        doc_buffer.extend(ids_list)

    while True:
        pos = 0
        row = np.empty(row_capacity, dtype=np.uint32)
        while pos < row_capacity:
            while len(doc_buffer) < buffer_size:
                try:
                    refill()
                except StopIteration:
                    break
            if not doc_buffer:
                return  # document stream exhausted

            remaining = row_capacity - pos
            # find largest doc that fits entirely
            best_idx, best_len = -1, 0
            for i, doc in enumerate(doc_buffer):
                dl = len(doc)
                if dl <= remaining and dl > best_len:
                    best_idx, best_len = i, dl
            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                row[pos:pos + best_len] = np.array(doc, dtype=np.uint32)
                pos += best_len
            else:
                # no doc fits: crop shortest buffered doc to fill exactly
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                row[pos:pos + remaining] = np.array(doc[:remaining], dtype=np.uint32)
                pos += remaining
        yield row


def tokenize_split(split, T, n_shards=-1):
    """Tokenize one split into .bin shards under the base dir. Returns list of paths."""
    tokenizer = get_tokenizer()
    out_dir = os.path.join(get_base_dir(), "data", split)
    os.makedirs(out_dir, exist_ok=True)

    parquet_paths = list_parquet_files()
    if split == "train":
        parquet_paths = parquet_paths[:-1]
        if n_shards > 0:
            parquet_paths = parquet_paths[:n_shards]
    else:
        parquet_paths = parquet_paths[-1:]
    assert parquet_paths, f"No parquet files for split {split}; download data first"

    print(f"Tokenizing split={split}: {len(parquet_paths)} parquet shards, T={T}")
    out_paths = []
    for filepath in parquet_paths:
        # read this shard's docs in full (row groups), feed as an iterator
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(filepath)
        texts_iter = [pf.read_row_group(rg_idx).column('text').to_pylist()
                      for rg_idx in range(pf.num_row_groups)]

        def local_iter():
            for texts in texts_iter:
                yield texts, (0, 0)

        rows = list(bestfit_pack(iter(local_iter()), tokenizer, T))
        if not rows:
            continue
        arr = np.stack(rows)  # (N, T+1) uint32
        out_path = os.path.join(out_dir, f"shard_{len(out_paths):05d}.bin")
        arr.tofile(out_path)
        out_paths.append(out_path)
        print(f"  {out_path}: {arr.shape[0]} rows, {arr.size/1e6:.2f}M tokens")
    print(f"Done: {len(out_paths)} .bin shards in {out_dir}")
    return out_paths


def list_bin_files(split):
    d = os.path.join(get_base_dir(), "data", split)
    if not os.path.exists(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize parquet shards to .bin")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--t", type=int, default=2048, help="sequence length T")
    parser.add_argument("--n-shards", type=int, default=-1, help="max parquet shards (train only)")
    args = parser.parse_args()
    tokenize_split(args.split, args.t, n_shards=args.n_shards)
