"""Summarise the teacher-labelled corpus into results/m2_summary.md.

    python code/m2_summary.py --tag full --sample data/samples/fnspid_clean_48040_seed42.csv

Regenerate rather than hand-editing, so the reported distribution cannot drift from the labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from m1_reproduce import default_results_dir
from teacher_label import DUP_SUFFIX, default_data_dir, labels_path


def load(tag: str, sample: Path, data_dir: Path):
    path = labels_path(data_dir, tag)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    repeats = int(df["uid"].str.endswith(DUP_SUFFIX).sum())
    df = df[~df["uid"].str.endswith(DUP_SUFFIX)]
    meta = pd.read_csv(sample, dtype=str)[["uid", "Date", "Stock_symbol", "Article_title"]]
    df = df.merge(meta, on="uid", how="inner")
    df["year"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.year
    return df, repeats


def pct_table(series: pd.Series, title: str) -> list[str]:
    counts = series.value_counts()
    lines = [f"### {title}", "", "| value | n | share |", "|---|---|---|"]
    for key, n in counts.items():
        lines.append(f"| `{key}` | {n:,} | {n / len(series) * 100:.1f}% |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--test-from", type=int, default=2019, help="first year of the held-out period")
    args = ap.parse_args()

    data_dir = args.data_dir or default_data_dir()
    results_dir = args.results_dir or default_results_dir()
    df, repeats = load(args.tag, args.sample, data_dir)

    titles = df["Article_title"].fillna("").str.lower()
    normalised = (titles.str.replace(r"[0-9]+", "#", regex=True)
                        .str.replace(r"[^a-z# ]", "", regex=True)
                        .str.replace(r"\s+", " ", regex=True).str.strip())

    test = df[df["year"] >= args.test_from]
    train = df[df["year"] < args.test_from]

    lines = [
        "# M2 - teacher-labelled news corpus",
        "",
        f"**{len(df):,} headlines** labelled by `claude-opus-5` at low reasoning effort, via the "
        f"Batch API, plus {repeats:,} repeated headlines used to measure teacher self-consistency.",
        f"Source: FNSPID, {int(df['year'].min())}-{int(df['year'].max())}, "
        f"{df['Stock_symbol'].nunique():,} tickers.",
        "",
        "Labels are not in this repository: they are regenerable from the sample and the code, and "
        "the raw corpus is large. See `code/teacher_label.py`.",
        "",
    ]
    lines += pct_table(df["category"], "Category")
    lines += pct_table(df["materiality"], "Materiality")
    lines += pct_table(df["direction"], "Direction")

    lines += [
        "### Corpus shape",
        "",
        "| property | value |",
        "|---|---|",
        f"| headlines | {len(df):,} |",
        f"| distinct tickers | {df['Stock_symbol'].nunique():,} |",
        f"| median headlines per ticker | {int(df['Stock_symbol'].value_counts().median())} |",
        f"| exact duplicate titles | {titles.duplicated().sum():,} ({titles.duplicated().mean() * 100:.1f}%) |",
        f"| duplicates after masking numbers | {normalised.duplicated().sum():,} "
        f"({normalised.duplicated().mean() * 100:.1f}%) |",
        "",
        "### Planned split",
        "",
        f"Time-based, so the test period is one the model never saw: train on years before "
        f"{args.test_from} ({len(train):,} rows), hold out {args.test_from} onward "
        f"({len(test):,} rows, {len(test) / len(df) * 100:.0f}%).",
        "",
        f"The held-out period contains {int((test['materiality'] == 'high').sum()):,} "
        "high-materiality headlines, which is the scarcest thing being measured and so the binding "
        "constraint on how precisely recall can be reported.",
        "",
    ]

    check = results_dir / "m2_earnings_check.json"
    if check.exists():
        c = json.loads(check.read_text(encoding="utf-8"))
        lines += [
            "### External validation of the teacher",
            "",
            "Fidelity only shows whether a student copies its teacher; it cannot catch a teacher "
            "that is confidently wrong. This check tests a label against a record kept elsewhere.",
            "",
            f"Of headlines tagged `earnings`, **{c['hit_rate_tagged'] * 100:.1f}%** fall within "
            f"{c['window_days']} days of a real quarterly report date, against "
            f"**{c['hit_rate_other'] * 100:.1f}%** for everything else - a **{c['lift']:.2f}x lift** "
            f"over the base rate, across {c['n_tagged']:,} tagged headlines.",
            "",
            "Reported as lift rather than a raw hit rate: earnings news is common enough here that "
            "a bare percentage would look convincing without demonstrating anything.",
            "",
        ]

    out = results_dir / "m2_summary.md"
    results_dir.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
