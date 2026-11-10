"""
Pretraining dataset: ClimbMix parquet shards (karpathy/climbmix-400b-shuffle).

Same dataset nanochat uses. Shards are ~100 MB compressed, ~250M chars each.
The last shard (6542) is always the validation shard.

Download:   python -m nanotern.dataset -n 12
"""
import os
import time
import argparse
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanotern.common import get_base_dir

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542  # last shard, pinned as validation
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
DATA_DIR = os.path.join(get_base_dir(), "data", "climbmix")


def list_parquet_files(data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        return []
    parquet_files = sorted(f for f in os.listdir(data_dir)
                           if f.endswith('.parquet') and not f.endswith('.tmp'))
    return [os.path.join(data_dir, f) for f in parquet_files]


def parquets_iter(split, tokenizer_batch_size=128):
    """
    Infinite-friendly iterator over document text batches from parquet.
    split: 'train' (all but last shard) or 'val' (last shard only).
    Yields (list_of_text, (pq_idx, rg_idx)).
    """
    assert split in ["train", "val"]
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, f"No parquet files found in {DATA_DIR}; run: python -m nanotern.dataset -n 12"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for pq_idx, filepath in enumerate(parquet_paths):
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            for i in range(0, len(texts), tokenizer_batch_size):
                yield texts[i:i + tokenizer_batch_size], (pq_idx, rg_idx)


def download_single_file(index):
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True
    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    os.remove(path)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                return False
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=12,
                        help="Number of train shards to download (validation shard always added)")
    parser.add_argument("-w", "--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    num_train_shards = min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)  # always download the validation shard
    print(f"Downloading {len(ids_to_download)} shards to {DATA_DIR}")
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)
    successful = sum(1 for r in results if r)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards")
