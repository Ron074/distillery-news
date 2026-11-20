"""M6 step 3 - does gating `high` on calibrated confidence recover the teacher's effect on returns?

    python code/m6_downstream.py --sample data/samples/fnspid_clean_48040_seed42.csv \
        --wrds-dir <your-wrds-pull>

M5 found the student's `high` tier carries 47% of the teacher's high-vs-low effect on same-day
abnormal returns, because it calls `high` about 2.4 times as often and the extra calls move less.
The obvious fix - raise the bar for `high` - can be made to "win" trivially: flag only the few most
extreme headlines and any effect gets large. So the bar is NOT tuned on returns. It is fixed before
any price is read, on the 2018 validation year alone:

    call `high` when the calibrated P(high) >= tau, where tau makes the student flag `high` on
    2018 exactly as often as the teacher labelled it there; otherwise the larger of low / medium.

That rule is then applied unchanged to the 2019+ test split and pushed through M5's own join and
regression (m5_downstream.build / fit): same rows, same prices, same controls, same clustering, plus
a shuffled-label null arm for the gated labels.

Reference prices are academically licensed, read from --wrds-dir, never copied into this
repository, and used for validation only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import softmax

from m1_reproduce import default_results_dir
from m5_downstream import fit, build


def gated_rule(rd: Path) -> dict:
    """Everything decided from validation data: calibrator, threshold, and what it implies."""
    cal = json.loads((rd / "m6_calibration.json").read_text(encoding="utf-8"))["heads"]["materiality"]
    fv, ft = np.load(rd / "logits_val.npz"), np.load(rd / "logits_test.npz")
    names = json.loads(str(fv["label_sets"]))["materiality"]
    k = names.index("high")
    # The calibrator with the lower validation NLL - chosen on 2018, before any test number.
    arm = min(("temperature", "temperature_bias"), key=lambda a: cal["arms"][a]["val"]["nll"])
    T = cal["temperature"] if arm == "temperature" else cal["temp_bias"]["temperature"]
    b = np.zeros(len(names)) if arm == "temperature" else np.array(cal["temp_bias"]["bias"])

    pv = softmax(fv["z_materiality"] / T + b, axis=1)
    pt = softmax(ft["z_materiality"] / T + b, axis=1)
    teacher_rate_val = float((fv["y_materiality"] == k).mean())
    tau = float(np.quantile(pv[:, k], 1 - teacher_rate_val))

    def label(p):
        other = np.where(p[:, names.index("medium")] > p[:, names.index("low")], "medium", "low")
        return np.where(p[:, k] >= tau, "high", other)

    return {"calibrator": arm, "temperature": T, "bias": b.tolist(), "tau": tau,
            "teacher_high_rate_val": teacher_rate_val,
            "student_high_rate_val_gated": float((label(pv) == "high").mean()),
            "test_uid": ft["uid"], "test_gated": label(pt),
            "test_teacher": np.array(names)[ft["y_materiality"]]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--wrds-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    rd = args.results_dir or default_results_dir()

    rule = gated_rule(rd)
    print(f"rule fixed on 2018: calibrator {rule['calibrator']}, tau = {rule['tau']:.4f}; "
          f"teacher calls high on {rule['teacher_high_rate_val']:.2%} of 2018, gated student "
          f"{rule['student_high_rate_val_gated']:.2%}")

    preds = pd.read_csv(rd / "preds_3head.csv", dtype=str)
    gated = pd.DataFrame({"uid": rule["test_uid"], "student_materiality_gated": rule["test_gated"],
                          "teacher_check": rule["test_teacher"]})
    merged = preds.merge(gated, on="uid", how="inner", validate="one_to_one")
    if len(merged) != len(preds) or (merged["teacher_check"] != merged["teacher_materiality"]).any():
        raise SystemExit("gated labels do not line up with preds_3head.csv row for row")
    gated_path = rd / "preds_m6_gated.csv"
    merged.drop(columns="teacher_check").to_csv(gated_path, index=False)

    j = build(SimpleNamespace(preds=gated_path, sample=args.sample, wrds_dir=args.wrds_dir,
                              market_adjust=True))
    runs = {
        "teacher": fit(j, "teacher_materiality", "teacher labels (claude-opus-5)"),
        "student": fit(j, "student_materiality", "student, argmax (as in M5)"),
        "student_gated": fit(j, "student_materiality_gated", "student, high gated on calibrated P(high)"),
        "student_gated_shuffled": fit(j, "student_materiality_gated", "gated labels shuffled (null arm)",
                                      seed=args.seed),
    }
    t_h = runs["teacher"]["terms"]["high"]["coef"]
    print(f"\n|abnormal return| % ~ tier + cap band, SEs clustered by ticker ({runs['teacher']['clusters']:,} tickers)")
    print(f"  {'arm':24s} {'n high':>7s} {'high vs low':>12s} {'se':>7s} {'t':>7s}  share of teacher")
    for key, r in runs.items():
        h = r["terms"]["high"]
        print(f"  {key:24s} {r['n_by_tier']['high']:>7,} {h['coef']:>+11.2f}pp {h['se']:>7.3f} "
              f"{h['t']:>7.2f}  {h['coef'] / t_h:>6.0%}")

    out = {
        "milestone": "M6 gated high, downstream", "measure": "|abnormal return| %",
        "population": "held-out test split only", "se": "CR1, clustered by ticker",
        "rule": {k: v for k, v in rule.items() if not k.startswith("test_")},
        "rule_fixed_on": "2018 validation year; no returns read before it was fixed",
        "n_joined": int(len(j)), "n_tickers": int(j["ticker"].nunique()),
        "share_of_teacher_effect": {k: round(r["terms"]["high"]["coef"] / t_h, 4) for k, r in runs.items()},
        "runs": runs,
    }
    (rd / "m6_downstream.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {rd / 'm6_downstream.json'}")


if __name__ == "__main__":
    main()
