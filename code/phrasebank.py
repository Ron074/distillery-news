"""Financial PhraseBank: download, load, and split.

The corpus is Malo et al. (2014), distributed as FinancialPhraseBank-v1.0.zip. Each line of the
Sentences_*.txt files is ``<sentence>@<label>`` in latin-1, where the file suffix is the share of
annotators who agreed on the label (50/66/75/all).

Usage:
    python code/phrasebank.py --agreement 50agree        # download + report class balance

Paths come from --data-dir or $DISTILLERY_DATA_DIR, defaulting to <repo>/data; raw files land in
<data-dir>/raw/ and are git-ignored.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ZIP_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
ZIP_NAME = "FinancialPhraseBank-v1.0.zip"

AGREEMENTS = {
    "50agree": "Sentences_50Agree.txt",
    "66agree": "Sentences_66Agree.txt",
    "75agree": "Sentences_75Agree.txt",
    "allagree": "Sentences_AllAgree.txt",
}

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}


def default_data_dir() -> Path:
    env = os.environ.get("DISTILLERY_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


def download(data_dir: Path) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / ZIP_NAME
    if not zip_path.exists():
        print(f"downloading {ZIP_URL} -> {zip_path}")
        req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "distillery-news"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as out:
            out.write(resp.read())
    extract_dir = raw_dir / "FinancialPhraseBank-v1.0"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    return extract_dir


def _find_sentences_file(extract_dir: Path, filename: str) -> Path:
    matches = sorted(extract_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {extract_dir}")
    return matches[0]


def load(agreement: str = "50agree", data_dir: Path | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns ``sentence``, ``label`` (name), ``label_id``."""
    if agreement not in AGREEMENTS:
        raise ValueError(f"agreement must be one of {sorted(AGREEMENTS)}")
    data_dir = data_dir or default_data_dir()
    extract_dir = download(data_dir)
    path = _find_sentences_file(extract_dir, AGREEMENTS[agreement])

    rows = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sentence, _, label = line.rpartition("@")
            rows.append((sentence.strip(), label.strip()))

    df = pd.DataFrame(rows, columns=["sentence", "label"])
    df = df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL2ID)
    if df["label_id"].isna().any():
        bad = df.loc[df["label_id"].isna(), "label"].unique()
        raise ValueError(f"unexpected labels in {path.name}: {bad}")
    df["label_id"] = df["label_id"].astype(int)
    return df


def split(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.1, test_frac: float = 0.1):
    """Deterministic stratified train/val/test split, stratified on label."""
    rng_parts = {"train": [], "val": [], "test": []}
    for _, group in df.groupby("label_id", sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        rng_parts["test"].append(group.iloc[:n_test])
        rng_parts["val"].append(group.iloc[n_test : n_test + n_val])
        rng_parts["train"].append(group.iloc[n_test + n_val :])
    out = {}
    for name, parts in rng_parts.items():
        merged = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out[name] = merged
    return out["train"], out["val"], out["test"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agreement", default="50agree", choices=sorted(AGREEMENTS))
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load(args.agreement, args.data_dir)
    train, val, test = split(df, seed=args.seed)
    print(f"\nFinancial PhraseBank ({args.agreement}): {len(df)} sentences")
    print("\nclass balance (full corpus):")
    print(df["label"].value_counts().rename("n").to_frame().assign(pct=lambda d: (d["n"] / len(df) * 100).round(1)))
    print(f"\nsplit (seed={args.seed}): train={len(train)}  val={len(val)}  test={len(test)}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        counts = part["label"].value_counts().reindex(LABELS).to_dict()
        print(f"  {name:5s} {counts}")
    print("\nexamples:")
    for _, row in df.head(3).iterrows():
        print(f"  [{row['label']:8s}] {row['sentence'][:100]}")


if __name__ == "__main__":
    main()
