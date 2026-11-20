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

# The one cost figure not read from a results file. The 24 Sep Batch API run (20,500 requests:
# 20,000 headlines plus 500 repeats, the 9-category prompt) was billed $23.26, computed by
# teacher_label.py collect from the batch's own usage and recorded in the project log at the time.
# Per-request usage was not saved, so this total is the record. It cross-checks: the 19 Nov
# real-time run (m4_teacher_live.json) costs 2.03x this per item, against the 2x the published
# 50% batch discount predicts.
BATCH_RUN_USD, BATCH_RUN_REQUESTS = 23.26, 20_500
# All of M2 - pilots, the 9-category run, the relabel after `analyst` was added, the consistency
# probe - was recorded as "about $79". Approximate; quoted only as such.
M2_TOTAL_USD_APPROX = 79


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
    live = json.loads((rd / "m4_teacher_live.json").read_text(encoding="utf-8"))
    cpu = json.loads((rd / "m3_checkpoint_check.json").read_text(encoding="utf-8"))
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
    ]

    # ---- (b) cost and speed --------------------------------------------------------------------
    batch_per_1000 = BATCH_RUN_USD / BATCH_RUN_REQUESTS * 1000
    rt_per_1000 = live["cost_per_1000_usd_warm_cache"]
    rt_ms = live["latency_ms"]
    stu_single, stu_batch = cpu["cpu_ms_per_headline_single"], cpu["cpu_ms_per_headline_batched"]
    labelled_rows = chk["n_train"] + chk["n_val"]
    min_label_cost = labelled_rows * batch_per_1000 / 1000

    def per_s(ms: float) -> str:
        return f"{1000 / ms:,.1f}" if ms >= 1 else f"{1000 / ms:,.0f}"

    lines += [
        "",
        "## Cost and speed per 1,000 headlines",
        "",
        "Every row names its device: the same student is 30x faster on a GPU than on this laptop, and the "
        "two figures had been quoted interchangeably. 'One at a time' is what a person typing a headline "
        "waits; 'batched' is throughput over a backlog.",
        "",
        "| arm | runs on | $ per 1,000 | ms per headline, one at a time | ms per headline, batched "
        "| headlines / s, one at a time | source |",
        "|---|---|---|---|---|---|---|",
        f"| teacher, real-time API | Anthropic API, over the network | **${rt_per_1000:.3f}** | "
        f"{rt_ms['median']:,.0f} (median; p90 {rt_ms['p90']:,.0f}) | - | {per_s(rt_ms['median'])} | "
        f"`m4_teacher_live.json`, {live['n_calls']} calls |",
        f"| teacher, Batch API | Anthropic batch | **${batch_per_1000:.3f}** | asynchronous: no "
        "per-request latency (batch wall-clock not recorded) | - | - | 24 Sep run, recorded at collection |",
        f"| student (FinBERT, 3 heads) | this laptop, CPU, {cpu['torch_threads']} threads | **$0** | "
        f"{stu_single:.1f} | {stu_batch:.1f} | {per_s(stu_single)} | `m3_checkpoint_check.json` |",
        f"| student (FinBERT, 3 heads) | Colab T4 GPU | $0 (free tier) | - | {m3['ms_per_headline']:.2f} "
        f"| - | `m3_3head.json` |",
        f"| TF-IDF + LR baseline | this laptop, CPU | **$0** | {base['ms_per_headline_single']:.2f} | "
        f"{base['ms_per_headline_batched']:.3f} | {per_s(base['ms_per_headline_single'])} | "
        "`m4_baseline.json` |",
        "",
        f"**Read across a row, one at a time:** the student answers in {stu_single:.0f} ms on a laptop "
        f"with no GPU and no key, about {rt_ms['median'] / stu_single:.0f}x faster than the teacher's "
        f"{rt_ms['median'] / 1000:.1f} s, at no marginal cost. The baseline is about "
        f"{stu_single / base['ms_per_headline_single']:.0f}x faster again - which is why the accuracy "
        "comparison above, not speed, is what decides between student and baseline.",
        "",
        f"The two teacher prices cross-check: real-time costs {rt_per_1000 / batch_per_1000:.2f}x the "
        "batch rate, where the published 50% batch discount predicts 2x. The real-time figure is with a "
        f"warm prompt cache ({live['tokens']['cache_read'] / (live['tokens']['cache_read'] + live['tokens']['input'] + live['tokens']['cache_write']):.0%} "
        f"of input tokens were cache reads); the first call of a session costs "
        f"${live['first_call_cost_usd']:.4f}, as it writes the cache.",
        "",
        "## The student is not free: it was paid for in labels",
        "",
        f"Its marginal cost is zero, but it exists because the teacher labelled its {labelled_rows:,} "
        f"training and validation headlines. At the measured batch rate that is **${min_label_cost:.2f}** - "
        "the cheapest those labels could have been bought. The project actually spent about "
        f"${M2_TOTAL_USD_APPROX} on M2, including pilots, a consistency probe, and relabelling the first "
        "20,000 headlines once the `analyst` category was added.",
        "",
        f"So against a real-time teacher the student pays for its labels after about "
        f"**{round(min_label_cost / rt_per_1000, 0):,.0f},000 headlines** at the minimum label cost, or "
        f"about {round(M2_TOTAL_USD_APPROX / rt_per_1000, 0):,.0f},000 at what was actually spent; every headline after "
        "that is free. Training itself cost nothing extra (a free Colab T4, minutes per run).",
        "",
        "## Caveats",
        "",
        f"- Teacher latency is wall-clock from this laptop, including the network, for `{live['model']}` at "
        f"`{live['effort']}` effort on {live['n_calls']} random test headlines, measured "
        f"{live['timestamp'][:10]} (UTC). It will vary with load and location; the median and p90 are reported "
        "for that reason.",
        "- The batch rate comes from the 9-category prompt; the final prompt is about 47% longer in input "
        "tokens, almost all of it cached. The real-time run uses the final prompt.",
        "- Live answers matched the stored teacher labels on "
        + ", ".join(f"{h} {v:.1%}" for h, v in live["agreement_with_stored_teacher_labels"].items())
        + ". That only confirms the same task was asked; the teacher's self-consistency is measured in M2.",
        "- Baseline timing varied up to 3x between runs on this laptop; medians of 5 repeats are reported.",
        "",
        "Sources: `m4_baseline.json`, `m4_teacher_live.json`, `m3_checkpoint_check.json`, "
        "`m3_3head.json`. Regenerate with `python code/m4_summary.py`.",
    ]
    out = rd / "m4_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
