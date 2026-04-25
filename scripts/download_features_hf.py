#!/usr/bin/env python
"""HuggingFace fallback downloader for SoccerNet Baidu features.

Why this exists: the official SoccerNet downloader pulls from
``data.soccer-net.org`` (KAUST). That host has been DNS-unreachable
since 2026-04-19 and the standard ``scripts/download_features.py
--features baidu`` path therefore fails. The same features are
mirrored on HuggingFace under
``OpenSportsLab/SoccerNet-ActionSpotting-Features`` with a slightly
different path convention (underscores instead of spaces in the match
directory name).

This script downloads the requested splits from the HF mirror and
writes them to the KAUST-style local paths that the rest of the
codebase already expects — so after the download completes, every
existing command works with ``--feature-type baidu`` unchanged.

Resumable: re-runs skip any file that already exists at the target
path. Use ``--force`` to re-download.

Tip: set ``HF_TOKEN`` in your environment for 10-20x faster throughput
(free token at huggingface.co/settings/tokens).
"""
import argparse
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "OpenSportsLab/SoccerNet-ActionSpotting-Features"
HF_PREFIX = "baidu_soccer_embeddings"


def hf_path_to_local(hf_path, local_base):
    """Map an HF file path to the local KAUST-convention path.

    HF:    baidu_soccer_embeddings/<split>/<league>/<year>/<match_underscores>/<file>
    KAUST: <local_base>/<league>/<year>/<match spaces>/<file>

    Split appears in the HF path but NOT the local path, to match the
    existing SoccerNet downloader behaviour.
    """
    parts = hf_path.split("/")
    if len(parts) < 5 or parts[0] != HF_PREFIX:
        return None
    split_name = parts[1]
    league = parts[2]
    year = parts[3]
    match_hf = parts[4]
    filename = "/".join(parts[5:]) if len(parts) > 5 else ""
    match_kaust = match_hf.replace("_", " ")
    return split_name, Path(local_base) / league / year / match_kaust / filename


_print_lock = threading.Lock()


def _download_one(hf_path, local_base, force):
    """Download one file to its KAUST-style target. Returns a status tag."""
    mapped = hf_path_to_local(hf_path, local_base)
    if mapped is None:
        return ("mapfail", hf_path, None)
    _, local_target = mapped
    if local_target.exists() and not force:
        return ("skip", hf_path, None)
    local_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                 filename=hf_path)
        shutil.copy2(cached, local_target)
        return ("ok", hf_path, None)
    except Exception as exc:
        return ("fail", hf_path, f"{type(exc).__name__}: {exc}")


def download_split(split, local_base, force=False, workers=10):
    """Download every Baidu file for a given split using a thread pool.

    Parallel IO helps a lot on the HF CDN: a single connection gets
    ~2.5 MB/s unauthenticated, but 10 concurrent requests together can
    push 15-25 MB/s even without an HF_TOKEN.
    """
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    split_prefix = f"{HF_PREFIX}/{split}/"
    split_files = [f for f in all_files
                   if f.startswith(split_prefix) and f.endswith(".npy")]
    print(f"[{split}] {len(split_files)} Baidu .npy files in the mirror; "
          f"{workers} workers.", flush=True)

    downloaded = skipped = failed = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one, f, local_base, force): f
                   for f in split_files}
        for i, fut in enumerate(as_completed(futures), start=1):
            status, hf_path, err = fut.result()
            if status == "ok":
                downloaded += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                with _print_lock:
                    print(f"  [{status}] {hf_path}: {err or ''}", flush=True)
            if i % 20 == 0 or i == len(split_files):
                elapsed = time.time() - t_start
                rate = downloaded / max(1.0, elapsed)
                remaining = len(split_files) - i
                eta_s = remaining / max(1e-3, rate) if rate > 0 else 0
                with _print_lock:
                    print(f"  [{split}] {i}/{len(split_files)} "
                          f"(downloaded={downloaded} skipped={skipped} "
                          f"failed={failed}) elapsed={elapsed:.0f}s "
                          f"rate={rate:.2f}/s eta={eta_s/60:.0f}m",
                          flush=True)
    return downloaded, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Baidu features via HuggingFace")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "valid", "test"],
                        choices=["train", "valid", "test", "challenge"],
                        help="Which splits to download. Defaults cover everything "
                             "needed for train+eval; add 'challenge' for submission.")
    parser.add_argument("--local-dir", default="data/",
                        help="KAUST-convention data root (same one used by the "
                             "rest of the pipeline). Defaults to data/.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download files that already exist locally.")
    parser.add_argument("--workers", type=int, default=10,
                        help="Thread-pool workers for parallel downloads "
                             "(default 10; raise for faster networks).")
    args = parser.parse_args()

    print(f"HF mirror: {REPO_ID}")
    print(f"Target:    {os.path.abspath(args.local_dir)}")
    if not os.getenv("HF_TOKEN"):
        print("[warn] HF_TOKEN not set. Throughput will be ~2.5 MB/s. "
              "Generate one at huggingface.co/settings/tokens and re-export.",
              flush=True)
    print()

    totals = [0, 0, 0]
    for split in args.splits:
        d, s, f = download_split(split, args.local_dir,
                                 force=args.force, workers=args.workers)
        totals[0] += d
        totals[1] += s
        totals[2] += f
        print()

    print("=" * 50)
    print(f"Total downloaded: {totals[0]}")
    print(f"Total skipped:    {totals[1]} (already present)")
    print(f"Total failed:     {totals[2]}")
    if totals[2] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
