"""Check the teacher's labels against external ground truth, with no human judgement involved.

The fidelity numbers elsewhere in this project only show whether the student copies the teacher. If
the teacher were confidently wrong, a perfect student would be confidently wrong too. This script
breaks that loop by testing a teacher label against a fact recorded somewhere else entirely.

    materiality  does the materiality tier track the size of the realised move, within cap bands?
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

# Compustat's quarterly report dates. The plan preferred edgar_earnings_dates.csv for being
# ticker-keyed with no join, but that file turned out to hold 13 tickers - it covered 10 of our
# 5,223. This one carries its own `tic` column too, so it needs no gvkey join either, and reaches
# 61% of our tickers across 1.2M announcement dates.
EARNINGS_FILE = "ccm_rdq.parquet"


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

    ann = pd.read_parquet(path, columns=["tic", "rdq"])
    ann = ann.rename(columns={"tic": "ticker", "rdq": "earnings_date"})
    ann["ticker"] = ann["ticker"].astype(str).str.upper().str.strip()
    ann["earnings_date"] = pd.to_datetime(ann["earnings_date"], errors="coerce").dt.normalize()
    ann = ann.dropna(subset=["ticker", "earnings_date"])
    ann = ann[ann["ticker"].isin(set(df["ticker"]))]

    if ann.empty:
        raise SystemExit("no announcement dates matched any ticker in the labeled set")

    # A headline counts as "on an announcement" if a real one falls within +/- window days.
    by_ticker: dict[str, pd.Series] = {t: g["earnings_date"] for t, g in ann.groupby("ticker")}
    window = pd.Timedelta(f"{int(args.window)}D")

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
    print(f"  tickers covered by rdq      {covered['ticker'].nunique():,} "
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


DAILY_FILE = "daily_stock_ciz3.parquet"
# Thousands of dollars, per the project plan's bands.
CAP_BANDS = [(0, 50_000, "nano <50M"), (50_000, 300_000, "micro"), (300_000, 2e6, "small"),
             (2e6, 1e7, "mid"), (1e7, 9e12, "large")]


def check_materiality(args) -> None:
    """Does the teacher's materiality tier track how much the stock actually moved?

    Category is close to objective; materiality is a judgement call, and self-consistency cannot
    tell a consistent judgement from a correct one. Only outside price data can.
    """
    from scipy.stats import mannwhitneyu

    wrds_dir = args.wrds_dir or default_wrds_dir()
    if not wrds_dir:
        raise SystemExit(f"pass --wrds-dir or set $DISTILLERY_WRDS_DIR to the folder holding {DAILY_FILE}")
    path = Path(wrds_dir) / DAILY_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found")

    data_dir = args.data_dir or default_data_dir()
    df = load_labeled(data_dir, args.tag, args.sample)

    daily = pd.read_parquet(path, columns=["Ticker", "DlyCalDt", "DlyRet", "DlyCap"])
    daily["ticker"] = daily["Ticker"].astype(str).str.upper().str.strip()
    daily["news_date"] = pd.to_datetime(daily["DlyCalDt"], errors="coerce").dt.normalize()
    daily = daily[daily["ticker"].isin(set(df["ticker"]))]

    j = df.merge(daily[["ticker", "news_date", "DlyRet", "DlyCap"]], on=["ticker", "news_date"])
    j["abs_ret"] = j["DlyRet"].astype(float).abs() * 100
    j["cap"] = j["DlyCap"].astype(float)
    j = j.dropna(subset=["abs_ret"])

    print(f"materiality check  (tag={args.tag})")
    print(f"  headlines joined to a trading day: {len(j):,} of {len(df):,}\n")
    overall = j.groupby("materiality")["abs_ret"].agg(["size", "median", "mean"])
    print("  median |return| on the news day, by tier:")
    for tier in ("low", "medium", "high"):
        if tier in overall.index:
            row = overall.loc[tier]
            print(f"    {tier:8s} n={int(row['size']):>6,}  median {row['median']:.2f}%  mean {row['mean']:.2f}%")

    print("\n  within market-cap band (the size control - small companies move more regardless):")
    print(f"    {'band':12s} {'low':>7s} {'medium':>7s} {'high':>7s} {'n_high':>7s}")
    bands = {}
    for lo, hi, name in CAP_BANDS:
        b = j[(j["cap"] >= lo) & (j["cap"] < hi)]
        if len(b) < args.min_band:
            continue
        med = b.groupby("materiality")["abs_ret"].median()
        n_high = int((b["materiality"] == "high").sum())
        bands[name] = {t: float(med.get(t, float("nan"))) for t in ("low", "medium", "high")} | {"n_high": n_high}
        print(f"    {name:12s} {med.get('low', float('nan')):>7.2f} {med.get('medium', float('nan')):>7.2f} "
              f"{med.get('high', float('nan')):>7.2f} {n_high:>7,}")

    high = j[j["materiality"] == "high"]["abs_ret"]
    low = j[j["materiality"] == "low"]["abs_ret"]
    _, p = mannwhitneyu(high, low, alternative="greater")
    print(f"\n  high vs low, Mann-Whitney one-sided p = {p:.2e}")
    print("  CAVEAT: same-day association, not causation, and repeated tickers are not clustered,")
    print("  so the p-value is optimistic. Reactive coverage is tagged low and moves a lot, which")
    print("  pushes against this result rather than for it.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "check": "materiality", "tag": args.tag, "n_joined": int(len(j)),
            "by_tier": {k: {"n": int(v["size"]), "median_abs_ret": float(v["median"]),
                            "mean_abs_ret": float(v["mean"])} for k, v in overall.iterrows()},
            "by_cap_band": bands, "mannwhitney_p_high_gt_low": float(p),
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="check", required=True)

    pm = sub.add_parser("materiality", help="does the materiality tier track the realised move?")
    pm.add_argument("--tag", required=True)
    pm.add_argument("--sample", type=Path, required=True)
    pm.add_argument("--wrds-dir", type=Path, default=None)
    pm.add_argument("--data-dir", type=Path, default=None)
    pm.add_argument("--min-band", type=int, default=200, help="skip cap bands thinner than this")
    pm.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("earnings", help="do 'earnings' tags land on real announcement dates?")
    p.add_argument("--tag", required=True, help="which label run to check")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--wrds-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--window", type=int, default=2, help="days either side that still counts as on-event")
    p.add_argument("--out", type=Path, default=None, help="write the numbers to this json")

    args = ap.parse_args()
    {"earnings": check_earnings, "materiality": check_materiality}[args.check](args)


if __name__ == "__main__":
    main()
