"""Download FNSPID and draw a reproducible random sample of headlines.

FNSPID's headline file is ~5.7 GB, so the sample is drawn in one streaming pass with random-key
reservoir sampling: every row gets an independent U(0,1) key and the n smallest keys win, which is
an exact uniform sample without ever holding the corpus in memory.

    python code/fnspid.py --n 50000            # download (once) + sample 50k rows

The sample keeps `Url` as a stable join key so M6 can recover article bodies later without
re-sampling, and records whether this file already carries a body for each row.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ID = "Zihan1004/FNSPID"
HEADLINE_FILE = "Stock_news/All_external.csv"
USECOLS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Article"]


def default_data_dir() -> Path:
    env = os.environ.get("DISTILLERY_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parent.parent / "data"


def download(data_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetching {REPO_ID}:{HEADLINE_FILE} (~5.7 GB, cached after the first run)")
    return Path(hf_hub_download(REPO_ID, HEADLINE_FILE, repo_type="dataset", local_dir=raw_dir))


def sample(csv_path: Path, n: int, seed: int, chunksize: int = 200_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    best_keys: np.ndarray | None = None
    best_rows: pd.DataFrame | None = None
    seen = kept_eligible = 0

    reader = pd.read_csv(csv_path, usecols=USECOLS, chunksize=chunksize, dtype=str,
                         on_bad_lines="skip", low_memory=False)
    for i, chunk in enumerate(reader, start=1):
        seen += len(chunk)
        chunk = chunk.dropna(subset=["Article_title", "Stock_symbol"])
        chunk = chunk[chunk["Article_title"].str.strip().str.len() > 0]
        if chunk.empty:
            continue
        kept_eligible += len(chunk)

        body = chunk["Article"].fillna("")
        chunk = chunk.drop(columns=["Article"])
        chunk["body_chars"] = body.str.len().astype(int)
        chunk["has_body"] = chunk["body_chars"] > 0

        keys = rng.random(len(chunk))
        if best_rows is None:
            best_keys, best_rows = keys, chunk
        else:
            best_keys = np.concatenate([best_keys, keys])
            best_rows = pd.concat([best_rows, chunk], ignore_index=True)
        if len(best_rows) > n:
            keep = np.argpartition(best_keys, n)[:n]
            best_keys, best_rows = best_keys[keep], best_rows.iloc[keep].reset_index(drop=True)
        if i % 10 == 0:
            print(f"  scanned {seen:,} rows ({kept_eligible:,} eligible)", flush=True)

    print(f"scanned {seen:,} rows, {kept_eligible:,} eligible, sampled {len(best_rows):,}")
    out = best_rows.sort_values("Date").reset_index(drop=True)
    out.insert(0, "uid", pd.util.hash_pandas_object(out["Url"].fillna(out["Article_title"]),
                                                    index=False).astype("uint64").map(lambda v: f"h{v:016x}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50_000, help="sample size to draw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None, help="use an already-downloaded FNSPID csv")
    args = ap.parse_args()

    data_dir = args.data_dir or default_data_dir()
    csv_path = args.csv or download(data_dir)

    df = sample(csv_path, args.n, args.seed)
    if df["uid"].duplicated().any():
        dupes = int(df["uid"].duplicated().sum())
        df = df.drop_duplicates(subset=["uid"]).reset_index(drop=True)
        print(f"dropped {dupes} rows sharing a uid (duplicate urls)")

    out_dir = data_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fnspid_sample_{len(df)}_seed{args.seed}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nwrote {out_path}")
    print(f"date range : {df['Date'].min()} -> {df['Date'].max()}")
    print(f"tickers    : {df['Stock_symbol'].nunique():,} distinct")
    print(f"with body  : {int(df['has_body'].sum()):,} of {len(df):,} rows carry article text here")
    print("\ntop publishers:")
    print(df["Publisher"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
