"""Load a saved student checkpoint, prove it is the M3 model, and re-measure it on this machine.

    python code/student_check.py --checkpoint <models dir>/stat_finbert_3head_<date> \
        --sample data/samples/fnspid_clean_48040_seed42.csv

The checkpoint lives outside the repository (see the models folder's README), so its path is an
argument and never appears in a source file.

What it checks, in order, before reporting anything:
1. student.pt hashes to the sha256 its own meta.json records - a truncated copy fails here.
2. meta.json's label sets and teacher-prompt hash match this repository's taxonomy.py, and its
   split counts and test scores match results/m3_3head.json. A checkpoint from another project, or
   trained on another prompt, fails here even though it would load and run.
3. The test split rebuilt from train_student.py's own functions has exactly the uids in
   results/preds_3head.csv.
4. Re-running the 8,591 test headlines on THIS machine's CPU reproduces the M3 accuracies, and the
   number of rows whose label differs from the GPU run's saved prediction is counted, not assumed
   to be zero.

Then it measures what only a local run can: CPU latency (batched and one headline at a time),
per-head confidence and its calibration against the teacher, and the share of headlines whose
least-confident field falls below a cutoff. Per-headline probabilities go to
results/preds_3head_probs.csv (git-ignored like every csv); the summary to
results/m3_checkpoint_check.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import taxonomy
from m1_reproduce import default_results_dir, load_tokenizer
from teacher_label import default_data_dir
from train_student import HEADS, Student, drop_near_duplicates, load_dataset, predict, split_by_time

CUTOFFS = (0.5, 0.6, 0.7, 0.8, 0.9)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_student(ckpt: Path, device: str = "cpu", verify: bool = True):
    """Return (model, tokenizer, heads, meta). Refuses a checkpoint whose weights do not hash to
    what its sidecar says, or whose label sets disagree with this repository's taxonomy."""
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    heads = json.loads((ckpt / "heads.json").read_text(encoding="utf-8"))
    if heads != {k: v for k, v in HEADS.items() if k in heads}:
        raise SystemExit(f"{ckpt.name}: label sets differ from taxonomy.py - not this project's model")
    if verify:
        got = sha256_of(ckpt / "student.pt")
        if got != meta["sha256"]:
            raise SystemExit(f"{ckpt.name}: student.pt hashes to {got[:16]}..., meta.json says "
                             f"{meta['sha256'][:16]}... - corrupt or mismatched copy")
    model = Student(meta["encoder"], heads)
    state = torch.load(ckpt / "student.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    tokenizer = load_tokenizer(str(ckpt))
    return model, tokenizer, heads, meta


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: |confidence - accuracy|, averaged over equal-width bins
    weighted by how many rows fall in each."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--labels", default="full")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--single-n", type=int, default=200,
                    help="headlines timed one at a time, for a per-request latency")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()

    results_dir = args.results_dir or default_results_dir()
    m3 = json.loads((results_dir / "m3_3head.json").read_text(encoding="utf-8"))

    # 1-2. The artifact itself.
    t0 = time.perf_counter()
    model, tokenizer, heads, meta = load_student(args.checkpoint)
    load_seconds = time.perf_counter() - t0
    prompt_sha1 = hashlib.sha1(taxonomy.SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
    problems = []
    if meta["prompt_sha1"] != prompt_sha1:
        problems.append(f"teacher prompt sha1 {meta['prompt_sha1']} != this repo's {prompt_sha1}")
    for k in ("n_train", "n_val", "n_test", "near_duplicates_dropped"):
        if meta[k] != m3[k]:
            problems.append(f"{k}: checkpoint {meta[k]} != M3 {m3[k]}")
    for h in heads:
        for metric in ("accuracy", "macro_f1"):
            if meta["test_scores"][h][metric] != m3["test"][h][metric]:
                problems.append(f"{h} {metric}: checkpoint run {meta['test_scores'][h][metric]} "
                                f"!= M3 {m3['test'][h][metric]}")
    if problems:
        raise SystemExit("checkpoint is not the M3 run on record:\n  " + "\n  ".join(problems))
    print(f"checkpoint {args.checkpoint.name}: sha256 {meta['sha256'][:16]}... verified; label sets, "
          f"prompt hash, split counts and test scores all match M3 (loaded in {load_seconds:.1f}s)")

    # 3. The rows.
    df = load_dataset(args.labels, args.sample, args.data_dir or default_data_dir())
    df, dropped = drop_near_duplicates(df)
    _, _, test = split_by_time(df, meta["split"]["val_year"], meta["split"]["test_from"])
    saved = pd.read_csv(results_dir / "preds_3head.csv", dtype=str)
    if set(saved["uid"]) != set(test["uid"]) or len(test) != meta["n_test"]:
        raise SystemExit("rebuilt test split does not match preds_3head.csv")
    saved = saved.set_index("uid").loc[test["uid"]]
    print(f"test split rebuilt: {len(test):,} rows, uids identical to preds_3head.csv")

    # 4. Re-run on this CPU.
    texts = test["Article_title"].tolist()
    threads = torch.get_num_threads()
    t0 = time.perf_counter()
    preds, probs = predict(model, tokenizer, texts, heads, "cpu", args.batch_size,
                           meta["max_length"], return_probs=True)
    batch_ms = (time.perf_counter() - t0) / len(texts) * 1000

    out = test[["uid"]].copy()
    per_head = {}
    for h, names in heads.items():
        labels = np.array([names[i] for i in preds[h]])
        teacher = test[h].to_numpy()
        correct = labels == teacher
        conf = probs[h].max(1)
        flips = int((labels != saved[f"student_{h}"].to_numpy()).sum())
        per_head[h] = {
            "accuracy_cpu": float(correct.mean()),
            "accuracy_m3": m3["test"][h]["accuracy"],
            "rows_differing_from_gpu_run": flips,
            "mean_confidence": float(conf.mean()),
            "ece_10bin": ece(conf, correct),
            "confident_minus_accurate": float(conf.mean() - correct.mean()),
        }
        out[f"student_{h}"] = labels
        out[f"teacher_{h}"] = teacher
        for j, label in enumerate(names):
            out[f"p_{h}_{label}"] = probs[h][:, j].round(6)
        out[f"conf_{h}"] = conf.round(6)
        print(f"  {h:12s} CPU acc {correct.mean():.4f} (M3 {m3['test'][h]['accuracy']:.4f})  "
              f"rows differing from GPU run: {flips}   mean conf {conf.mean():.3f}  "
              f"ECE {per_head[h]['ece_10bin']:.3f}")

    least = np.min(np.stack([probs[h].max(1) for h in heads], axis=1), axis=1)
    below = {str(c): float((least < c).mean()) for c in CUTOFFS}
    # Does low confidence pick out the rows the student gets wrong? Rate of any-field error
    # among headlines below each cutoff vs above it.
    any_wrong = np.zeros(len(test), dtype=bool)
    for h in heads:
        any_wrong |= out[f"student_{h}"].to_numpy() != out[f"teacher_{h}"].to_numpy()
    err_by_cut = {str(c): {"below": float(any_wrong[least < c].mean()) if (least < c).any() else None,
                           "at_or_above": float(any_wrong[least >= c].mean())} for c in CUTOFFS}
    print("  least-confident field below cutoff: "
          + "  ".join(f"{c}: {v:.1%}" for c, v in below.items()))
    print("  any-field error rate, below vs at/above: "
          + "  ".join(f"{c}: {v['below']:.1%}/{v['at_or_above']:.1%}" for c, v in err_by_cut.items()))

    # Latency one headline at a time, as the demo issues them. Warm up first.
    single = texts[: args.single_n]
    predict(model, tokenizer, single[:5], heads, "cpu", 1, meta["max_length"])
    t0 = time.perf_counter()
    for t in single:
        predict(model, tokenizer, [t], heads, "cpu", 1, meta["max_length"])
    single_ms = (time.perf_counter() - t0) / len(single) * 1000
    print(f"  CPU latency: {batch_ms:.1f} ms/headline batched ({args.batch_size}), "
          f"{single_ms:.1f} ms one at a time, {threads} threads")

    out.to_csv(results_dir / "preds_3head_probs.csv", index=False)
    run = {
        "milestone": "M3 checkpoint check (STAT side)",
        "checkpoint": args.checkpoint.name,
        "sha256": meta["sha256"],
        "checks_passed": ["sha256", "label sets", "prompt sha1", "split counts",
                          "test scores equal m3_3head.json", "test uids equal preds_3head.csv"],
        "n_test": len(test),
        "heads": per_head,
        "least_confident_field_below": below,
        "any_field_error_rate_by_cutoff": err_by_cut,
        "cpu_ms_per_headline_batched": batch_ms,
        "cpu_ms_per_headline_single": single_ms,
        "batch_size": args.batch_size, "single_n": len(single),
        "torch_threads": threads, "torch_version": torch.__version__,
        "cpu": platform.processor(),
        "predictions_file": "preds_3head_probs.csv",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.machine()} python {platform.python_version()}",
    }
    (results_dir / "m3_checkpoint_check.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nwrote {results_dir / 'm3_checkpoint_check.json'} and preds_3head_probs.csv")


if __name__ == "__main__":
    main()
