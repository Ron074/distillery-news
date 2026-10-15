"""Score the hand-labelled gold sheet against the teacher.

Fidelity says whether the student copied its teacher. It cannot say whether the teacher was
right. This is the only measurement in the project that can, because it compares the teacher
against a judgement formed without it.

    python code/score_gold.py --gold data/gold/gold_blank_pilot.csv --tag full

Reports raw agreement and Cohen's kappa. Kappa matters here: `materiality` is 73% `low`, so a
labeller who wrote `low` on every row would score 73% agreement while demonstrating nothing.
Kappa subtracts the agreement two labellers would reach by chance given how often each of them
uses each label, so the floor is 0 rather than the majority share.

Rows left blank are skipped. Rows marked `unsure` are reported separately and excluded from the
scores - which rows were hard is a finding, not a failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import taxonomy
from m1_reproduce import default_results_dir
from teacher_label import DUP_SUFFIX, default_data_dir, labels_path

UNSURE = "unsure"
ALLOWED = {"category": taxonomy.CATEGORIES,
           "materiality": taxonomy.MATERIALITY,
           "direction": taxonomy.DIRECTION}


def load_gold(path: Path, fields: list[str]) -> tuple[pd.DataFrame, dict]:
    """Read the filled sheet, normalise the entries, and refuse to guess at anything unrecognised."""
    df = pd.read_csv(path, dtype=str).fillna("")
    if "uid" not in df.columns:
        raise SystemExit(f"{path} has no uid column; it cannot be joined to the teacher's labels")

    for field in fields:
        if field not in df.columns:
            raise SystemExit(f"{path} has no `{field}` column (columns: {list(df.columns)})")
        # Case and stray whitespace are the labeller's typing, not disagreement with the teacher.
        df[field] = df[field].str.strip().str.lower()

        bad = sorted(set(df[field]) - set(ALLOWED[field]) - {"", UNSURE})
        if bad:
            # Dropping these silently would quietly shrink the sample and inflate agreement, which
            # is the exact failure mode this project keeps finding. Stop instead.
            raise SystemExit(
                f"unrecognised {field} values in {path.name}: {bad}\n"
                f"  allowed: {', '.join(ALLOWED[field])} (or '{UNSURE}', or blank for not yet done)\n"
                "  Fix the spelling in the sheet and run again."
            )

    filled = df[[all(df.loc[i, f] != "" for f in fields) for i in df.index]]
    counts = {
        "rows in sheet": len(df),
        "rows filled": len(filled),
        "rows blank": len(df) - len(filled),
    }
    for field in fields:
        counts[f"{field} marked unsure"] = int((filled[field] == UNSURE).sum())
    return filled, counts


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Chance-corrected agreement. Written out rather than imported so the arithmetic is visible."""
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = float((a.values == b.values).mean())
    # Chance agreement: the rate two labellers would match at if each kept their own label
    # frequencies but assigned them independently of the headline.
    expected = sum((a == label).mean() * (b == label).mean() for label in labels)
    if expected >= 1.0:
        return float("nan")  # both labellers used a single label throughout; kappa is undefined
    return (observed - expected) / (1 - expected)


def score_field(gold: pd.DataFrame, field: str) -> dict:
    usable = gold[gold[field] != UNSURE]
    if usable.empty:
        return {"n": 0}
    mine, teacher = usable[field], usable[f"teacher_{field}"]
    agree = mine.values == teacher.values

    # Where the two differ, grouped by what the human said - this is what tells you whether the
    # teacher has a systematic bias or just noise.
    confusion = (usable[~agree].groupby([field, f"teacher_{field}"]).size()
                 .sort_values(ascending=False).head(10))
    return {
        "n": int(len(usable)),
        "agreement": round(float(agree.mean()), 4),
        "kappa": round(cohen_kappa(mine, teacher), 4),
        "majority_if_always_most_common": round(float(teacher.value_counts(normalize=True).max()), 4),
        "top_disagreements": [{"annotator": a, "teacher": t, "n": int(n)}
                              for (a, t), n in confusion.items()],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, required=True, help="the filled-in gold sheet")
    ap.add_argument("--tag", default="full", help="label tag to compare against")
    ap.add_argument("--fields", default="category,materiality")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--annotator", default="the second annotator",
                    help="who labelled the sheet; appears in the printout and the saved report")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    data_dir = args.data_dir or default_data_dir()
    gold, counts = load_gold(args.gold, fields)

    print(f"gold sheet: {args.gold}")
    for key, value in counts.items():
        print(f"  {key:28s} {value}")
    if gold.empty:
        raise SystemExit("\nnothing filled in yet - label the sheet first, then run this")

    rows = [json.loads(l) for l in labels_path(data_dir, args.tag).read_text(encoding="utf-8").splitlines() if l.strip()]
    teacher = pd.DataFrame(rows)
    teacher = teacher[~teacher["uid"].str.endswith(DUP_SUFFIX)]
    teacher = teacher.rename(columns={f: f"teacher_{f}" for f in fields})

    before = len(gold)
    gold = gold.merge(teacher[["uid"] + [f"teacher_{f}" for f in fields]], on="uid", how="inner")
    if len(gold) < before:
        print(f"\n  {before - len(gold)} filled rows have no teacher label under tag '{args.tag}' "
              "and cannot be compared")
    if gold.empty:
        raise SystemExit(f"no overlap between the sheet and tag '{args.tag}'")

    report = {"milestone": "gold", "gold_sheet": str(args.gold), "labels_tag": args.tag,
              "annotator": args.annotator, "counts": counts, "fields": {}}

    print(f"\nthe teacher against {args.annotator}, labelling independently of it, "
          f"on {len(gold)} headlines:")
    print(f"\n  {'field':12s} {'n':>5s} {'agree':>8s} {'kappa':>8s} {'always-majority':>16s}")
    for field in fields:
        s = score_field(gold, field)
        report["fields"][field] = s
        if not s["n"]:
            print(f"  {field:12s} {'0':>5s}   every row marked unsure")
            continue
        print(f"  {field:12s} {s['n']:>5d} {s['agreement']:>7.1%} {s['kappa']:>8.3f} "
              f"{s['majority_if_always_most_common']:>15.1%}")

    for field in fields:
        s = report["fields"][field]
        if s.get("top_disagreements"):
            print(f"\nwhere {args.annotator} and the teacher disagreed on {field}:")
            for d in s["top_disagreements"]:
                print(f"  {d['n']:>3d}  annotator said {d['annotator']:<10s} "
                      f"teacher said {d['teacher']}")

    print("\nRead the kappa, not the agreement: the `always-majority` column is what a labeller")
    print("who ignored every headline and wrote the most common label would have scored.")
    print("Disagreement here is not automatically the teacher being wrong - it is the two of you")
    print("reading the same headline differently, and the boundary cases are the interesting part.")

    out = args.out or (args.results_dir or default_results_dir()) / "gold_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
