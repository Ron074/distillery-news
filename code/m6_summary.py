"""Write results/m6_summary.md from the M6 runs.

    python code/m6_summary.py

Reads m6_calibration.json, and m6_downstream.json if it exists (it needs the licensed price data,
so it may be run later, by someone with access). Regenerate rather than hand-edit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from m1_reproduce import default_results_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    rd = args.results_dir or default_results_dir()
    c = json.loads((rd / "m6_calibration.json").read_text(encoding="utf-8"))
    H = c["heads"]

    lines = [
        "# M6 - confidence: calibrate it, then route on it",
        "",
        "M6 was planned as headline against full article. It was reframed (see the project log) to ask "
        "where the extra context is worth paying for: leave headlines the student is sure of alone, and "
        "send the rest to the teacher. That needs confidence that means something, so calibration comes "
        f"first. Everything is fitted on the 2018 validation year ({c['n_val']:,} headlines) and judged on "
        f"the 2019+ test split ({c['n_test']:,}), against the teacher's labels.",
        "",
        "## Calibration",
        "",
        "| head | fitted temperature | test ECE: raw | temperature | temperature + class bias "
        "| test accuracy: raw | temperature + class bias |",
        "|---|---|---|---|---|---|---|",
    ]
    for h, v in H.items():
        a = v["arms"]
        lines.append(f"| `{h}` | {v['temperature']:.3f} | {a['raw']['test']['ece']:.3f} | "
                     f"{a['temperature']['test']['ece']:.3f} | {a['temperature_bias']['test']['ece']:.3f} | "
                     f"{a['raw']['test']['accuracy']:.4f} | {a['temperature_bias']['test']['accuracy']:.4f} |")
    m = H["materiality"]["arms"]
    raw_h, tb_h = m["raw"]["test"]["high"], m["temperature_bias"]["test"]["high"]
    lines += [
        "",
        "**Every head is over-confident** - each fitted temperature is above 1 - and dividing by it cuts "
        "calibration error on test without changing a single prediction.",
        "",
        "**The class-bias correction raises accuracy, and that is not a free improvement.** It moves the "
        "operating point back toward the common classes, undoing what class weighting did on purpose. "
        f"On `high` materiality, recall falls from {raw_h['recall']:.3f} to {tb_h['recall']:.3f} while "
        f"precision rises from {raw_h['precision']:.3f} to {tb_h['precision']:.3f}. These are two "
        "operating points of one model, chosen after training from 2018 data. The higher accuracies "
        "are not comparable with M4's baseline, which was class-weighted too and would gain from the "
        "same correction.",
        "",
        f"Before calibration the student's `high` calls carried a mean P(high) of "
        f"{raw_h['mean_p_high_when_flagged']:.3f} at precision {raw_h['precision']:.3f}. With the bias "
        f"correction the calibration error of P(high) itself falls from {raw_h['class_ece_high']:.3f} to "
        f"{tb_h['class_ece_high']:.3f}.",
        "",
        "## Routing: what a little teacher buys",
        "",
        "A headline's confidence is its least confident field. Below a cutoff it goes to the teacher's "
        "real-time API. Cutoffs were set on 2018 to escalate a target share and applied unchanged to "
        f"2019+. Cost is that share times the measured ${c['teacher_usd_per_1000_realtime']:.3f} per "
        "1,000.",
        "",
        "An escalated headline is not scored as certainly right. The teacher, asked again, agrees with "
        f"its own stored label only {c['teacher_self_consistency']['all_three']:.1%} of the time on all "
        "three fields, so escalated rows count at that rate ('adjusted'). That is still optimistic: the "
        "headlines the student finds hard are plausibly the ones the teacher is least consistent on. "
        "'Upper' counts them as certain.",
        "",
    ]
    for arm, title in (("temperature", "the model as trained (keeps `high` recall)"),
                       ("temperature_bias", "with the class-bias correction (higher accuracy, lower `high` recall)")):
        lines += [f"**{title}**", "",
                  "| escalated, test | teacher $ per 1,000 | all three fields agree: adjusted | upper "
                  "| student all three right, on the escalated rows |",
                  "|---|---|---|---|---|"]
        for r in c["routing"][arm]:
            esr = r["escalated_student_all_three_right"]
            lines.append(f"| {r['escalated_share_test']:.1%} | ${r['teacher_usd_per_1000']:.3f} | "
                         f"{r['all_three_adjusted']:.1%} | {r['all_three_upper_bound']:.1%} | "
                         f"{'-' if esr is None else f'{esr:.1%}'} |")
        lines.append("")
    r0, r5 = c["routing"]["temperature"][0], c["routing"]["temperature"][1]
    lines += [
        f"**Confidence finds the student's mistakes.** On the {r5['escalated_share_test']:.1%} of "
        f"headlines it is least sure of, the student gets all three fields right "
        f"{r5['escalated_student_all_three_right']:.1%} of the time, against {r0['kept_all_three_right']:.1%} "
        "overall. The 2018 cutoffs escalate almost exactly the intended share in 2019 and 2020, so they "
        "transfer across years.",
        "",
        "**The teacher's self-consistency is not a hard ceiling.** "
        + (lambda top: f"With the class-bias correction at {top['escalated_share_test']:.0%} escalation, "
           f"adjusted agreement reaches {top['all_three_adjusted']:.1%}, above the teacher's own repeat "
           f"rate of {c['teacher_self_consistency']['all_three']:.1%}. ")(c["routing"]["temperature_bias"][-1])
        + "That is not a leak. Self-consistency is how often two random draws of the teacher agree; a "
        "deterministic model that always gave the teacher's most likely answer would match a single "
        "stored draw more often than that. The M3 'share of reachable gap' figures, which treat "
        "self-consistency as the ceiling, therefore slightly overstate how close the student is.",
        "",
    ]

    dpath = rd / "m6_downstream.json"
    if dpath.exists():
        d = json.loads(dpath.read_text(encoding="utf-8"))
        rule, runs, share = d["rule"], d["runs"], d["share_of_teacher_effect"]
        lines += [
            "## Does a calibrated `high` recover the effect on returns?",
            "",
            f"M5 found the student's `high` carries part of the teacher's effect on same-day abnormal "
            "returns, diluted because it calls `high` too often. Here `high` is called only when the "
            f"calibrated P(high) is at least {rule['tau']:.3f} - a bar set on 2018 so the student flags "
            f"`high` exactly as often as the teacher did there ({rule['teacher_high_rate_val']:.2%}), "
            "before any return was read. Same join, controls and clustering as M5; "
            f"{d['n_joined']:,} headlines, {d['n_tickers']:,} tickers.",
            "",
            "| labels | headlines called `high` | high vs low, pp | se | t | share of teacher's effect |",
            "|---|---|---|---|---|---|",
        ]
        for k, r in runs.items():
            hh = r["terms"]["high"]
            lines.append(f"| {r['label_source']} | {r['n_by_tier']['high']:,} | {hh['coef']:+.2f} | "
                         f"{hh['se']:.3f} | {hh['t']:.2f} | {share[k]:.0%} |")
        lines += ["", "A same-day association, not causation; see M5 for why that caveat cuts against "
                  "the result rather than for it.", ""]
    else:
        lines += [
            "## Does a calibrated `high` recover the effect on returns? (not yet run)",
            "",
            "The gating rule is fixed in `code/m6_downstream.py` from 2018 data alone. Running it needs "
            "the licensed price data, so the result is pending; nothing about it is reported here until "
            "it has run.",
            "",
        ]
    lines += ["Sources: `m6_calibration.json`" + (", `m6_downstream.json`" if dpath.exists() else "")
              + ". Regenerate with `python code/m6_summary.py`."]
    (rd / "m6_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rd / 'm6_summary.md'}")


if __name__ == "__main__":
    main()
