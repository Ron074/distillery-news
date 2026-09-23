"""Check the teacher's labels against external ground truth, with no human judgement involved.

The fidelity numbers elsewhere in this project only show whether the student copies the teacher. If
the teacher were confidently wrong, a perfect student would be confidently wrong too. This script
breaks that loop by testing a teacher label against a fact recorded somewhere else entirely.

    earnings   headlines tagged `earnings` should fall on or beside a real earnings-announcement
               date for that ticker. Compares the hit rate for tagged items against the rate for
               everything else, so the result is a lift over the base rate rather than a bare number.

    python code/validate_teacher.py earnings --tag pilot --wrds-dir R:/Databases2

The reference data is academically licensed. It is read from --wrds-dir (or $DISTILLERY_WRDS_DIR),
never copied into this repository, and is used for validation only - it is never a training input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

EARNINGS_FILE = "edgar_earnings_dates.csv"


def default_data_dir() -> Path:
    env = os.environ.get("DISTILLERY_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parent.parent / "data"


def default_wrds_dir() -> Path | None:
    env = os.environ.get("DISTILLERY_WRDS_DIR")
    return Path(env) if env else None


def load_labeled(data_dir: Path, tag: str, sample: Path) -> pd.DataFrame:
    labels_path = data_dir / "labels" / f"teacher_{tag}.jsonl"
    if not labels_path.exists():
        raise SystemExit(f"no labels at {labels_path}; run the labeling step first")
    rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = pd.DataFrame(rows)
    labels = labels[~labels["uid"].str.endswith("_dup")]

    meta = pd.read_csv(sample, dtype=str)[["uid", "Date", "Stock_symbol"]]
    df = labels.merge(meta, on="uid", how="inner")
    df["news_date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df["ticker"] = df["Stock_symbol"].str.upper().str.strip()
    return df.dropna(subset=["news_date", "ticker"])


def check_earnings(args) -> None:
    wrds_dir = args.wrds_dir or default_wrds_dir()
    if not wrds_dir:
        raise SystemExit("pass --wrds-dir or set $DISTILLERY_WRDS_DIR to the folder holding "
                         f"{EARNINGS_FILE}")
    path = Path(wrds_dir) / EARNINGS_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found")

    data_dir = args.data_dir or default_data_dir()
    df = load_labeled(data_dir, args.tag, args.sample)

    ann = pd.read_csv(path, usecols=["ticker", "earnings_date"], dtype=str)
    ann["ticker"] = ann["ticker"].str.upper().str.strip()
    ann["earnings_date"] = pd.to_datetime(ann["earnings_date"], errors="coerce").dt.normalize()
    ann = ann.dropna(subset=["ticker", "earnings_date"])
    ann = ann[ann["ticker"].isin(set(df["ticker"]))]

    if ann.empty:
        raise SystemExit("no announcement dates matched any ticker in the labeled set")

    # A headline counts as "on an announcement" if a real one falls within +/- window days.
    by_ticker: dict[str, pd.Series] = {t: g["earnings_date"] for t, g in ann.groupby("ticker")}
    window = pd.Timedelta(days=args.window)

    def near(row) -> bool | None:
        dates = by_ticker.get(row["ticker"])
        if dates is None:
            return None  # ticker absent from the reference file; not evidence either way
        return bool(((dates - row["news_date"]).abs() <= window).any())

    df["near_announcement"] = df.apply(near, axis=1)
    covered = df[df["near_announcement"].notna()].copy()
    if covered.empty:
        raise SystemExit("none of the labeled tickers appear in the announcement file")

    tagged = covered[covered["category"] == "earnings"]
    other = covered[covered["category"] != "earnings"]
    if tagged.empty:
        raise SystemExit("the labeled set contains no headlines tagged 'earnings'")

    hit_tagged = tagged["near_announcement"].mean()
    hit_other = other["near_announcement"].mean() if len(other) else float("nan")
    lift = hit_tagged / hit_other if hit_other else float("nan")

    print(f"earnings-tag check  (tag={args.tag}, window=+/-{args.window}d)")
    print(f"  labeled headlines           {len(df):,}")
    print(f"  tickers covered by EDGAR    {covered['ticker'].nunique():,} "
          f"({len(covered):,} of {len(df):,} headlines)")
    print(f"  tagged 'earnings'           {len(tagged):,}")
    print()
    print(f"  on a real announcement, tagged earnings : {hit_tagged:6.1%}")
    print(f"  on a real announcement, everything else : {hit_other:6.1%}")
    print(f"  lift                                    : {lift:6.2f}x")
    print()
    if len(tagged) < 30:
        print("  CAUTION: fewer than 30 tagged headlines - indicative only, too few to quote.")
    print("  Reading: a high rate with little lift means earnings news is simply common in this")
    print("  corpus. Lift well above 1 is the evidence that the tag tracks the real event.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "check": "earnings", "tag": args.tag, "window_days": args.window,
            "n_labeled": int(len(df)), "n_covered": int(len(covered)), "n_tagged": int(len(tagged)),
            "hit_rate_tagged": float(hit_tagged), "hit_rate_other": float(hit_other),
            "lift": float(lift),
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="check", required=True)

    p = sub.add_parser("earnings", help="do 'earnings' tags land on real announcement dates?")
    p.add_argument("--tag", required=True, help="which label run to check")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--wrds-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--window", type=int, default=2, help="days either side that still counts as on-event")
    p.add_argument("--out", type=Path, default=None, help="write the numbers to this json")

    args = ap.parse_args()
    {"earnings": check_earnings}[args.check](args)


if __name__ == "__main__":
    main()
