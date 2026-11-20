"""M6 step 2 - calibrate the student on 2018, then price confidence routing on 2019+.

    python code/m6_calibrate.py

Reads results/logits_{val,test}.npz (from m6_logits.py), results/m4_teacher_live.json (the
teacher's real-time price) and the teacher's measured self-consistency. Writes
results/m6_calibration.json. Nothing here touches prices or returns.

Two calibrators per head, both fitted by minimising negative log-likelihood against the teacher's
labels on the 2018 validation year only:
  * temperature - one scalar; rescales confidence, never changes a prediction.
  * temperature + per-class bias - can also undo a shift in one class's scores, which is what class
    weighting plausibly did to `high`. It CAN change predictions, so its accuracy is reported too.

Routing: a headline's confidence is its least confident field. Below a cutoff it goes to the
teacher. Cutoffs are chosen on 2018 to escalate a target share, then applied unchanged to test.
An escalated headline is NOT counted as certainly right: re-asked, the teacher agrees with its own
stored label only at its self-consistency rate. Both the naive upper bound and the
self-consistency-adjusted figure are reported, and the adjusted one is still optimistic, because
headlines the student finds hard are plausibly ones the teacher is less consistent on too.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax, softmax

from m1_reproduce import default_results_dir
from m3_summary import TEACHER_SELF_CONSISTENCY, TEACHER_SELF_CONSISTENCY_ALL_THREE

ESCALATE_SHARES = (0.0, 0.05, 0.10, 0.20, 0.33, 0.50)


def load(rd: Path, split: str):
    f = np.load(rd / f"logits_{split}.npz", allow_pickle=False)
    heads = json.loads(str(f["label_sets"]))
    return f, heads


def nll(z, y) -> float:
    return float(-log_softmax(z, axis=1)[np.arange(len(y)), y].mean())


def fit_temperature(z, y) -> float:
    r = minimize_scalar(lambda lt: nll(z / np.exp(lt), y), bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def fit_temp_bias(z, y) -> tuple[float, np.ndarray]:
    k = z.shape[1]

    def f(p):
        b = np.concatenate([[0.0], p[1:]])           # first class fixed: biases are relative
        return nll(z / np.exp(p[0]) + b, y)

    r = minimize(f, np.zeros(k), method="L-BFGS-B")
    return float(np.exp(r.x[0])), np.concatenate([[0.0], r.x[1:]])


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = p.max(1), p.argmax(1)
    correct = pred == y
    idx = np.clip((conf * bins).astype(int), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def class_ece(pk: np.ndarray, is_k: np.ndarray, bins: int = 10) -> float:
    """Calibration of one class's probability across all rows: when the model says 30% `high`,
    are 30% of those headlines `high`?"""
    idx = np.clip((pk * bins).astype(int), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(pk[idx == b].mean() - is_k[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def describe(p, y, names) -> dict:
    out = {"accuracy": float((p.argmax(1) == y).mean()), "nll": float(-np.log(
        np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()),
        "ece": ece(p, y), "mean_conf": float(p.max(1).mean())}
    if "high" in names:
        k = names.index("high")
        flag, true = p.argmax(1) == k, y == k
        out["high"] = {
            "flagged": int(flag.sum()), "true": int(true.sum()),
            "recall": float((flag & true).sum() / true.sum()),
            "precision": float((flag & true).sum() / flag.sum()) if flag.any() else None,
            "mean_p_high_when_flagged": float(p[flag, k].mean()) if flag.any() else None,
            "class_ece_high": class_ece(p[:, k], true),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    rd = args.results_dir or default_results_dir()

    fv, heads = load(rd, "val")
    ft, heads_t = load(rd, "test")
    if heads != heads_t or str(fv["checkpoint_sha256"]) != str(ft["checkpoint_sha256"]):
        raise SystemExit("val and test logits come from different checkpoints or label sets")
    live = json.loads((rd / "m4_teacher_live.json").read_text(encoding="utf-8"))
    teacher_per_1000 = live["cost_per_1000_usd_warm_cache"]

    calib, probs = {}, {"val": {}, "test": {}}
    for h, names in heads.items():
        zv, yv = fv[f"z_{h}"], fv[f"y_{h}"]
        zt, yt = ft[f"z_{h}"], ft[f"y_{h}"]
        T = fit_temperature(zv, yv)
        Tb, b = fit_temp_bias(zv, yv)
        arms = {"raw": (1.0, np.zeros(len(names))), "temperature": (T, np.zeros(len(names))),
                "temperature_bias": (Tb, b)}
        calib[h] = {"temperature": T, "temp_bias": {"temperature": Tb, "bias": b.tolist()}, "arms": {}}
        for arm, (t, bias) in arms.items():
            pv, pt = softmax(zv / t + bias, axis=1), softmax(zt / t + bias, axis=1)
            calib[h]["arms"][arm] = {"val": describe(pv, yv, names), "test": describe(pt, yt, names)}
            probs["val"].setdefault(arm, {})[h] = pv
            probs["test"].setdefault(arm, {})[h] = pt
        a = calib[h]["arms"]
        print(f"{h:12s} T={T:.3f}  test ECE raw {a['raw']['test']['ece']:.3f} -> temp "
              f"{a['temperature']['test']['ece']:.3f} -> temp+bias {a['temperature_bias']['test']['ece']:.3f}"
              f"   acc {a['raw']['test']['accuracy']:.4f} / {a['temperature_bias']['test']['accuracy']:.4f}")
        if "high" in a["raw"]["test"]:
            for arm in a:
                hi = a[arm]["test"]["high"]
                print(f"    high [{arm:16s}] flagged {hi['flagged']:4d} (teacher {hi['true']})  "
                      f"recall {hi['recall']:.3f}  precision {hi['precision']:.3f}  "
                      f"mean P(high) when flagged {hi['mean_p_high_when_flagged']:.3f}  "
                      f"class-ECE {hi['class_ece_high']:.3f}")

    # ---- routing --------------------------------------------------------------------------------
    sc_field = TEACHER_SELF_CONSISTENCY
    routing = {}
    # raw and temperature share predictions (temperature only rescales); temperature_bias moves the
    # operating point back toward the majority classes, so it is reported as a separate arm.
    for arm in ("raw", "temperature", "temperature_bias"):
        cv = np.min(np.stack([probs["val"][arm][h].max(1) for h in heads], 1), 1)
        ct = np.min(np.stack([probs["test"][arm][h].max(1) for h in heads], 1), 1)
        right_t = {h: probs["test"][arm][h].argmax(1) == ft[f"y_{h}"] for h in heads}
        all_right = np.logical_and.reduce([right_t[h] for h in heads])
        rows = []
        for share in ESCALATE_SHARES:
            cut = float(np.quantile(cv, share)) if share > 0 else 0.0   # chosen on 2018 only
            esc = ct < cut
            keep = ~esc
            e = float(esc.mean())
            row = {"target_share_val": share, "cutoff": cut, "escalated_share_test": e,
                   "teacher_usd_per_1000": e * teacher_per_1000,
                   "kept_all_three_right": float(all_right[keep].mean()) if keep.any() else None,
                   "escalated_student_all_three_right": float(all_right[esc].mean()) if esc.any() else None}
            # Headline-level agreement with the stored teacher labels, all three fields at once.
            row["all_three_upper_bound"] = float(all_right[keep].sum() + esc.sum()) / len(ct)
            row["all_three_adjusted"] = float(all_right[keep].sum()
                                              + esc.sum() * TEACHER_SELF_CONSISTENCY_ALL_THREE) / len(ct)
            row["per_field_adjusted"] = {h: float(right_t[h][keep].sum() + esc.sum() * sc_field[h]) / len(ct)
                                         for h in heads}
            rows.append(row)
        routing[arm] = rows

    for arm in ("temperature", "temperature_bias"):
        print(f"\nrouting, {arm} (cutoffs set on 2018, applied to 2019+):")
        print(f"  {'target':>6s} {'escalated':>9s} {'$/1000':>7s} {'all-3 adjusted':>14s} {'upper':>7s} "
              f"{'student right on escalated':>27s}")
        for r in routing[arm]:
            esr = r["escalated_student_all_three_right"]
            print(f"  {r['target_share_val']:>6.0%} {r['escalated_share_test']:>9.1%} "
                  f"{r['teacher_usd_per_1000']:>7.3f} {r['all_three_adjusted']:>14.1%} "
                  f"{r['all_three_upper_bound']:>7.1%} {('-' if esr is None else f'{esr:.1%}'):>27s}")

    out = {
        "milestone": "M6 calibration and routing",
        "checkpoint_sha256": str(ft["checkpoint_sha256"]),
        "fit_on": "2018 validation year, NLL against teacher labels", "judged_on": "2019+ test split",
        "n_val": int(len(fv["uid"])), "n_test": int(len(ft["uid"])),
        "heads": calib, "routing": routing,
        "teacher_usd_per_1000_realtime": teacher_per_1000,
        "teacher_self_consistency": {"all_three": TEACHER_SELF_CONSISTENCY_ALL_THREE, **sc_field},
        "routing_note": "escalated rows scored at the teacher's average self-consistency, which is "
                        "optimistic for hard headlines; the upper bound scores them as certain",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (rd / "m6_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {rd / 'm6_calibration.json'}")


if __name__ == "__main__":
    main()
