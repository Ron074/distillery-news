"""M4 (a) - a TF-IDF + logistic-regression baseline on the distillation task itself.

M1 already has a TF-IDF number, but it is PhraseBank sentiment - a different task, different labels,
different data - so it says nothing about whether the 110M-parameter student earns its cost here.
This trains the same kind of baseline on exactly the student's rows: same labels, same near-duplicate
filter, same temporal split, one classifier per head.

    python code/m4_baseline.py --sample data/samples/fnspid_clean_48040_seed42.csv

The split is not rebuilt by hand. It comes from train_student.py's own functions and is then checked
against the M3 run on record - row counts from results/m3_3head.json, and the exact set of test uids
from results/preds_3head.csv. A baseline scored on even slightly different rows is not a baseline.

Fairness choices, so the comparison is not rigged in the student's favour:
- class_weight="balanced", the same inverse-frequency idea the student's loss uses; without it the
  baseline would simply never predict `high` materiality and look worse for the wrong reason.
- the regularisation strength C is chosen per head on the 2018 validation year, the same year the
  student used to pick its epoch. Test rows are touched once.
- a shuffled-label arm, mirroring the student's null test. It must collapse.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from m1_reproduce import default_results_dir
from teacher_label import default_data_dir
from train_student import HEADS, drop_near_duplicates, load_dataset, split_by_time

C_GRID = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0)
N_BOOT = 2000


def check_split_matches_m3(train, val, test, dropped, results_dir: Path) -> dict:
    """Fail loudly if these are not the rows the student was trained and tested on."""
    m3 = json.loads((results_dir / "m3_3head.json").read_text(encoding="utf-8"))
    want = {"n_train": m3["n_train"], "n_val": m3["n_val"], "n_test": m3["n_test"],
            "near_duplicates_dropped": m3["near_duplicates_dropped"]}
    got = {"n_train": len(train), "n_val": len(val), "n_test": len(test),
           "near_duplicates_dropped": int(dropped)}
    if got != want:
        raise SystemExit(f"split does not match the M3 run on record:\n  M3   {want}\n  here {got}")

    preds = pd.read_csv(results_dir / "preds_3head.csv", dtype=str)
    m3_uids, our_uids = set(preds["uid"]), set(test["uid"])
    if m3_uids != our_uids:
        raise SystemExit(f"test uids differ from M3's: {len(m3_uids - our_uids)} only in M3, "
                         f"{len(our_uids - m3_uids)} only here")
    # Same rows is not enough if the labels on them moved; check the teacher labels agree too.
    merged = test.merge(preds, on="uid", how="inner", validate="one_to_one")
    mismatched = {h: int((merged[h] != merged[f"teacher_{h}"]).sum()) for h in HEADS}
    if any(mismatched.values()):
        raise SystemExit(f"teacher labels differ from M3's predictions file: {mismatched}")
    return {**got, "test_uids_identical_to_m3": True, "teacher_labels_identical_to_m3": True,
            "student_preds": preds}


def score(truth, pred, names) -> dict:
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=names, average="macro", zero_division=0)),
        "per_class": classification_report(truth, pred, labels=names, output_dict=True,
                                           zero_division=0),
    }


def paired_cluster_bootstrap(truth, a, b, clusters, names, n_boot, seed) -> dict:
    """Student minus baseline on the same rows, resampling whole tickers rather than rows.

    Headlines about one company are correlated, so a row bootstrap would overstate precision -
    the same reason M5 clusters its standard errors by ticker."""
    truth, a, b = (np.asarray(x) for x in (truth, a, b))
    codes, uniq = pd.factorize(np.asarray(clusters))
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    high = names.index("high") if "high" in names else None

    def stats(idx):
        t, pa, pb = truth[idx], a[idx], b[idx]
        out = {"accuracy": (pa == t).mean() - (pb == t).mean(),
               "macro_f1": f1_score(t, pa, labels=names, average="macro", zero_division=0)
                           - f1_score(t, pb, labels=names, average="macro", zero_division=0)}
        if high is not None:
            hit = t == "high"
            out["high_recall"] = (pa[hit] == "high").mean() - (pb[hit] == "high").mean() if hit.any() else np.nan
        return out

    point = stats(np.arange(len(truth)))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        s = stats(np.concatenate([members[k] for k in pick]))
        for k, v in s.items():
            draws[k].append(v)
    return {k: {"diff": float(point[k]),
                "ci95": [float(np.nanpercentile(draws[k], 2.5)), float(np.nanpercentile(draws[k], 97.5))]}
            for k in point} | {"n_clusters": len(uniq), "n_boot": n_boot}


def baseline_at_student_precision(clf, Xte, truth, student) -> dict:
    """The baseline's best `high` recall at a precision no worse than the student's.

    The two models sit at different operating points (the student flags more, so it catches more
    and is right less often), so comparing their recall at their own defaults partly compares
    thresholds. Here the baseline's threshold is chosen ON THE TEST SET to maximise recall subject
    to matching the student's precision - deliberately generous to the baseline, so any gap that
    survives is not a threshold artifact. Threshold-free comparison (average precision) needs the
    student's probabilities, which the M3 run did not keep."""
    p_high = clf.predict_proba(Xte)[:, list(clf.classes_).index("high")]
    hit = truth == "high"
    s_flag = student == "high"
    s_prec = float((hit & s_flag).sum() / s_flag.sum())
    s_rec = float((hit & s_flag).sum() / hit.sum())
    order = np.argsort(-p_high)
    tp = np.cumsum(hit[order])
    k = np.arange(1, len(order) + 1)
    prec, rec = tp / k, tp / hit.sum()
    ok = prec >= s_prec
    best = int(np.argmax(np.where(ok, rec, -1))) if ok.any() else None
    return {"student_precision": s_prec, "student_recall": s_rec,
            "baseline_recall": float(rec[best]) if best is not None else 0.0,
            "baseline_precision": float(prec[best]) if best is not None else None,
            "baseline_flagged": int(best + 1) if best is not None else 0,
            "student_flagged": int(s_flag.sum()),
            "threshold_chosen_on": "test (favours the baseline)"}


def fit_head(Xtr, ytr, C, seed):
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)
    return clf.fit(Xtr, ytr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="full")
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--val-year", type=int, default=2018)
    ap.add_argument("--test-from", type=int, default=2019)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timing-n", type=int, default=1000,
                    help="headlines timed one at a time, for a per-request latency")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()

    data_dir = args.data_dir or default_data_dir()
    results_dir = args.results_dir or default_results_dir()

    df = load_dataset(args.labels, args.sample, data_dir)
    df, dropped = drop_near_duplicates(df)
    train, val, test = split_by_time(df, args.val_year, args.test_from)
    check = check_split_matches_m3(train, val, test, dropped, results_dir)
    student_preds = check.pop("student_preds")
    print(f"split verified against M3: train {len(train):,}  val {len(val):,}  test {len(test):,}  "
          f"(near-duplicates dropped {dropped:,}); test uids and teacher labels identical")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    t0 = time.perf_counter()
    Xtr = vec.fit_transform(train["Article_title"])
    vec_seconds = time.perf_counter() - t0
    Xva = vec.transform(val["Article_title"])
    Xte = vec.transform(test["Article_title"])
    print(f"tfidf vocabulary {len(vec.vocabulary_):,} terms, fit on train only")

    rng = np.random.default_rng(args.seed)
    heads_out, models, train_seconds = {}, {}, vec_seconds
    for head, names in HEADS.items():
        # C on the validation year only.
        val_f1 = {}
        for C in C_GRID:
            clf = fit_head(Xtr, train[head], C, args.seed)
            val_f1[C] = float(f1_score(val[head], clf.predict(Xva), labels=names,
                                       average="macro", zero_division=0))
        best_C = max(val_f1, key=val_f1.get)

        t0 = time.perf_counter()
        clf = fit_head(Xtr, train[head], best_C, args.seed)
        train_seconds += time.perf_counter() - t0
        models[head] = clf
        base_pred = clf.predict(Xte)
        real = score(test[head], base_pred, names)

        # Null arm: identical pipeline and C, training labels permuted.
        shuffled = rng.permutation(train[head].to_numpy())
        null = score(test[head], fit_head(Xtr, shuffled, best_C, args.seed).predict(Xte), names)

        counts = test[head].value_counts()
        majority = {"label": counts.idxmax(), "share": float(counts.max() / len(test))}

        sp = student_preds.set_index("uid").loc[test["uid"]]
        m3_student = score(sp[f"teacher_{head}"], sp[f"student_{head}"], names)
        paired = paired_cluster_bootstrap(test[head], sp[f"student_{head}"].to_numpy(), base_pred,
                                          test["Stock_symbol"], names, N_BOOT, args.seed)
        matched = None
        if head == "materiality":
            matched = baseline_at_student_precision(clf, Xte, test[head].to_numpy(),
                                                    sp[f"student_{head}"].to_numpy())
            print(f"               high at the student's precision {matched['student_precision']:.3f}: "
                  f"baseline recall {matched['baseline_recall']:.3f} vs student "
                  f"{matched['student_recall']:.3f} (threshold picked on test, favouring the baseline)")
        heads_out[head] = {
            "high_at_matched_precision": matched,
            "C_grid_val_macro_f1": {str(k): v for k, v in val_f1.items()},
            "chosen_C": best_C,
            "test": real,
            "null_test": null,
            "majority_on_test": majority,
            "student_m3_recomputed_from_preds": {k: m3_student[k] for k in ("accuracy", "macro_f1")},
            "student_minus_baseline": paired,
        }
        print(f"  {head:12s} C={best_C:<5} acc {real['accuracy']:.4f}  macro-F1 {real['macro_f1']:.4f}"
              f"   | shuffled {null['accuracy']:.4f}/{null['macro_f1']:.4f}"
              f"   | majority {majority['share']:.4f}   | student {m3_student['accuracy']:.4f}")
        print("               student - baseline: " + "  ".join(
            f"{k} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
            for k, v in paired.items() if isinstance(v, dict)), flush=True)

    # Timing is noisy on a laptop (two otherwise identical runs differed 3x), so take the median
    # of several repeats rather than one pass.
    texts = test["Article_title"].tolist()
    sample = texts[: args.timing_n]

    def batched():
        t0 = time.perf_counter()
        X = vec.transform(texts)
        for clf in models.values():
            clf.predict(X)
        return (time.perf_counter() - t0) / len(texts) * 1000

    def single():  # one headline per call, as a demo or a live feed would issue them
        t0 = time.perf_counter()
        for t in sample:
            x = vec.transform([t])
            for clf in models.values():
                clf.predict(x)
        return (time.perf_counter() - t0) / len(sample) * 1000

    batch_runs = [batched() for _ in range(5)]
    single_runs = [single() for _ in range(5)]
    batch_ms, single_ms = float(np.median(batch_runs)), float(np.median(single_runs))

    hi = heads_out["materiality"]["test"]["per_class"]["high"]
    print(f"\n  high materiality: recall {hi['recall']:.3f}  precision {hi['precision']:.3f}  "
          f"(support {int(hi['support'])})")
    print(f"  CPU: {batch_ms:.4f} ms/headline batched, {single_ms:.3f} ms/headline one at a time "
          f"(median of 5)")

    run = {
        "milestone": "M4", "arm": "tfidf(1,2) + logistic regression, one per head",
        "labels_tag": args.labels, "seed": args.seed,
        "split": {"train_before": args.val_year, "val_year": args.val_year, "test_from": args.test_from},
        "split_check": check,
        "tfidf_vocabulary": len(vec.vocabulary_),
        "heads": heads_out,
        "train_seconds": train_seconds,
        "ms_per_headline_batched": batch_ms,
        "ms_per_headline_single": single_ms,
        "timing_n_single": len(sample),
        "timing_repeats_ms": {"batched": batch_runs, "single": single_runs},
        "device": "cpu",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.machine()} python {platform.python_version()}",
    }
    out = results_dir / "m4_baseline.json"
    out.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
