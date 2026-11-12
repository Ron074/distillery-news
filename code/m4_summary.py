"""Write results/m4_summary.md from the M4 runs.

    python code/m4_summary.py

Regenerate rather than hand-edit, so the reported numbers cannot drift from the runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from m1_reproduce import default_results_dir
from m3_summary import TEACHER_SELF_CONSISTENCY, headroom


def ci(d: dict, scale: float = 1.0, digits: int = 3) -> str:
    lo, hi = d["ci95"]
    return f"{d['diff'] * scale:+.{digits}f} [{lo * scale:+.{digits}f}, {hi * scale:+.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    rd = args.results_dir or default_results_dir()

    base = json.loads((rd / "m4_baseline.json").read_text(encoding="utf-8"))
    m3 = json.loads((rd / "m3_3head.json").read_text(encoding="utf-8"))
    heads = base["heads"]
    chk = base["split_check"]

    lines = [
        "# M4 - is the student worth its cost? A TF-IDF baseline on the distillation task",
        "",
        "M1's TF-IDF number is PhraseBank sentiment and says nothing about this task. This is the same "
        "kind of baseline - TF-IDF unigrams and bigrams into one logistic regression per head - trained on "
        f"**exactly the student's rows**: {chk['n_train']:,} train, {chk['n_val']:,} validation (2018), "
        f"{chk['n_test']:,} test (2019 onward), after the same {chk['near_duplicates_dropped']:,} "
        "near-duplicates were dropped. The script checks this rather than assuming it: row counts against "
        "`m3_3head.json`, the test uid set against `preds_3head.csv`, and the teacher label on every test "
        "row against the same file. All three matched.",
        "",
        "Set up not to lose on a technicality: balanced class weights (the student's loss is "
        "inverse-frequency weighted too), and the regularisation strength C chosen per head on the 2018 "
        "validation year from "
        f"{', '.join(heads['category']['C_grid_val_macro_f1'])} - the chosen values "
        f"({', '.join(f'{h} {v['chosen_C']:g}' for h, v in heads.items())}) are all inside the grid, not "
        "on its edge.",
        "",
        "## Student against baseline, same 8,591 test rows",
        "",
        "| head | majority | TF-IDF + LR | student (M3) | student - baseline, accuracy [95% CI] "
        "| share of reachable gap: baseline / student |",
        "|---|---|---|---|---|---|",
    ]
    for h, v in heads.items():
        b, s = v["test"]["accuracy"], m3["test"][h]["accuracy"]
        lines.append(
            f"| `{h}` | {v['majority_on_test']['share']:.3f} | {b:.4f} | {s:.4f} | "
            f"{ci(v['student_minus_baseline']['accuracy'])} | {headroom(b, h):.0%} / {headroom(s, h):.0%} |")
    lines += [
        "",
        "| head | baseline macro-F1 | student macro-F1 | student - baseline [95% CI] |",
        "|---|---|---|---|",
    ]
    for h, v in heads.items():
        lines.append(f"| `{h}` | {v['test']['macro_f1']:.4f} | {m3['test'][h]['macro_f1']:.4f} | "
                     f"{ci(v['student_minus_baseline']['macro_f1'])} |")

    mat = heads["materiality"]
    bh = mat["test"]["per_class"]["high"]
    sh = m3["test"]["materiality"]["per_class"]["high"]
    mp = mat["high_at_matched_precision"]
    boot = mat["student_minus_baseline"]
    lines += [
        "",
        f"Intervals resample whole tickers ({boot['n_clusters']:,} of them, {boot['n_boot']:,} draws), "
        "not rows - headlines about one company are correlated, the same reason M5 clusters by ticker. "
        "The ceiling for 'share of reachable gap' is the teacher's self-consistency "
        f"({', '.join(f'{h} {TEACHER_SELF_CONSISTENCY[h]:.1%}' for h in heads)}).",
        "",
        "## `high` materiality, the class the project exists for",
        "",
        "| | recall | precision | flagged |",
        "|---|---|---|---|",
        f"| student (M3) | **{sh['recall']:.3f}** | {sh['precision']:.3f} | {mp['student_flagged']} |",
        f"| baseline, default threshold | {bh['recall']:.3f} | {bh['precision']:.3f} | "
        f"{round(bh['recall'] * bh['support'] / bh['precision']) if bh['precision'] else 0} |",
        f"| baseline, best threshold at the student's precision | {mp['baseline_recall']:.3f} | "
        f"{mp['baseline_precision']:.3f} | {mp['baseline_flagged']} |",
        "",
        f"Support: {int(sh['support'])} `high` headlines in test. The two models sit at different "
        "operating points by default, so the third row gives the baseline its best possible threshold at "
        "the student's precision, **chosen on the test set itself** - an advantage the student did not "
        f"get. The student still catches {(sh['recall'] - mp['baseline_recall']) * 100:.0f} percentage "
        "points more of the market-movers. A threshold-free comparison (average precision) needs the student's "
        "probabilities, which the M3 run did not keep.",
        "",
        "## Null arm",
        "",
        "| head | baseline | labels shuffled | majority |",
        "|---|---|---|---|",
    ]
    for h, v in heads.items():
        lines.append(f"| `{h}` | {v['test']['accuracy']:.4f} | {v['null_test']['accuracy']:.4f} | "
                     f"{v['majority_on_test']['share']:.4f} |")
    lines += [
        "",
        "Same pipeline and C, training labels permuted. Every head collapses to or below the majority "
        "rate (below, because balanced weighting spreads guesses across rare classes), so the baseline's "
        "scores come from the headlines, not from the split.",
        "",
        "## What it means",
        "",
        "- **The student's margin is narrower than the M3 table alone suggests.** On category accuracy a "
        "linear model on word counts gets within "
        f"{heads['category']['student_minus_baseline']['accuracy']['diff'] * 100:.1f} points; on materiality accuracy "
        "the two are indistinguishable. Much of the teacher's judgement on these two fields is "
        "recoverable from vocabulary.",
        "- **Where the student earns its cost is the hard part:** direction "
        f"({heads['direction']['student_minus_baseline']['accuracy']['diff'] * 100:+.1f} points accuracy, "
        f"{heads['direction']['student_minus_baseline']['macro_f1']['diff']:+.3f} macro-F1), and the rare "
        f"`high` tier ({sh['recall']:.0%} recall against the baseline's best {mp['baseline_recall']:.0%} at "
        "equal precision). A plausible reading, not yet tested: direction turns on word order and "
        "negation, and `high` on finding a rare class at all - both places a bag of words is weak. An error "
        "analysis of the rows where the two models disagree would check it.",
        "- Materiality accuracy is the wrong lens for this project: it is "
        f"{mat['majority_on_test']['share']:.0%} `low`, so both models score on the easy majority.",
        "",
        f"Baseline speed on this laptop's CPU, all three heads: {base['ms_per_headline_batched']:.3f} ms per "
        f"headline batched, {base['ms_per_headline_single']:.2f} ms one at a time (median of 5 repeats; "
        "single runs varied up to 3x). The full cost table - teacher, student, baseline - is the second half "
        "of M4.",
        "",
        f"Source: `m4_baseline.json` ({base['timestamp']}, {base['platform']}). "
        "Regenerate with `python code/m4_summary.py`.",
    ]
    out = rd / "m4_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
