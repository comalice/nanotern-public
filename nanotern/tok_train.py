"""
Train the BPE tokenizer on ClimbMix text.

Usage:
    python -m nanotern.tok_train --max-chars 1000000000 --vocab-size 32768
"""
import os
import argparse

import torch

from nanotern.common import get_base_dir
from nanotern.dataset import parquets_iter
from nanotern.tokenizer import RustBPETokenizer


def iter_texts(max_chars, doc_cap=10_000, extra_text_files=()):
    """Yield documents for BPE training: any extra text files first (small
    slices, e.g. symbolic corpora — PLAN piece 13), then ClimbMix train
    documents, until max_chars total."""
    total = 0
    for path in extra_text_files:
        with open(path) as f:
            for block in f.read().split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                total += len(block)
                yield block[:doc_cap]
        print(f"Mixed in extra text file {path} (running total {total/1e6:.1f}M chars)")
    for texts, _ in parquets_iter("train"):
        for t in texts:
            t = t[:doc_cap]
            total += len(t)
            yield t
            if total >= max_chars:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the BPE tokenizer")
    parser.add_argument("--max-chars", type=int, default=1_000_000_000)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--doc-cap", type=int, default=10_000)
    parser.add_argument("--extra-text-file", action="append", default=[],
                        help="extra plain-text file (blank-line separated docs) mixed into BPE training")
    args = parser.parse_args()

    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    print(f"Training BPE tokenizer: {args.max_chars/1e6:.0f}M chars, vocab {args.vocab_size}")
    tok = RustBPETokenizer.train_from_iterator(
        iter_texts(args.max_chars, args.doc_cap, args.extra_text_file), args.vocab_size)
    print(f"Trained. Vocab size: {tok.get_vocab_size()}")
    tok.save(tokenizer_dir)

    # token bytes table (for bpb evaluation): number of bytes per token id
    token_bytes = torch.zeros(args.vocab_size, dtype=torch.int32)
    for i in range(args.vocab_size):
        token_bytes[i] = len(tok.decode([i]).encode("utf-8"))
    torch.save(token_bytes, os.path.join(tokenizer_dir, "token_bytes.pt"))

    # sanity roundtrip
    sample = "The quick brown fox jumps over the lazy dog. 12345 émojis 🎉"
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"Roundtrip test: {sample!r} -> {len(ids)} tokens -> {back[:60]!r}...")
    assert back == sample, "Roundtrip mismatch!"
    print("Token bytes table saved. Done.")
