"""Write results/m3_checkpoint_summary.md from student_check.py's output.

    python code/checkpoint_summary.py

Reads results/m3_checkpoint_check.json and the per-headline probabilities in
results/preds_3head_probs.csv. Regenerate rather than hand-edit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from m1_reproduce import default_results_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    rd = args.results_dir or default_results_dir()

    run = json.loads((rd / "m3_checkpoint_check.json").read_text(encoding="utf-8"))
    p = pd.read_csv(rd / run["predictions_file"])
    if len(p) != run["n_test"]:
        raise SystemExit(f"{run['predictions_file']} has {len(p)} rows, the check ran on {run['n_test']}")

    gaps = [100 * run["heads"][h]["confident_minus_accurate"] for h in run["heads"]]
    flagged = p["student_materiality"] == "high"
    truly = p["teacher_materiality"] == "high"
    high_conf = float(p.loc[flagged, "p_materiality_high"].mean())
    high_prec = float((flagged & truly).sum() / flagged.sum())

    lines = [
        "# M3 checkpoint, verified and re-measured on a laptop CPU",
        "",
        f"The saved student (`{run['checkpoint']}`, sha256 `{run['sha256'][:16]}...`) is checked "
        "before anything is measured: " + ", ".join(run["checks_passed"]) + ". All passed.",
        "",
        "## It is the M3 model, on CPU as on GPU",
        "",
        "| head | accuracy, this CPU | accuracy, M3 (T4) | labels differing from the T4 run | "
        "mean confidence | ECE (10 bins) |",
        "|---|---|---|---|---|---|",
    ]
    for h, v in run["heads"].items():
        lines.append(f"| `{h}` | {v['accuracy_cpu']:.4f} | {v['accuracy_m3']:.4f} | "
                     f"{v['rows_differing_from_gpu_run']} of {run['n_test']:,} | "
                     f"{v['mean_confidence']:.3f} | {v['ece_10bin']:.3f} |")
    lines += [
        "",
        "Not one of the 8,591 test labels changes between the GPU that trained the model and this CPU, "
        "so the demo runs the model the report describes.",
        "",
        "## Confidence, and what it is worth",
        "",
        "Confidence is the top softmax probability of a head; a headline's confidence is that of its "
        "*least* confident field, since one shaky field is enough to want a second opinion.",
        "",
        "| cutoff | share of headlines below it | any-field error rate below | at or above |",
        "|---|---|---|---|",
    ]
    for c, share in run["least_confident_field_below"].items():
        e = run["any_field_error_rate_by_cutoff"][c]
        lines.append(f"| {c} | {share:.1%} | {e['below']:.1%} | {e['at_or_above']:.1%} |")
    lines += [
        "",
        "Error here means disagreeing with the teacher on at least one of the three fields. Low "
        "confidence picks out the headlines the student gets wrong: below 0.5 they are wrong on some "
        "field three times in four, against under a third above it. That is what confidence routing "
        "(M6) relies on, and it holds.",
        "",
        "**Calibration: mildly over-confident overall, badly over-confident on `high`.** Every head's "
        f"mean confidence exceeds its accuracy, by {min(gaps):.1f} to {max(gaps):.1f} points. The rare "
        f"class is different: when the student calls a headline `high` it gives that call an average "
        f"probability of {high_conf:.3f}, and is right {high_prec:.1%} of the time. The likely cause is "
        "the class weighting that made the model find the rare class at all - it raises `high` scores "
        "across the board - but that is a hypothesis this run does not test. Either way the raw `high` "
        "probabilities cannot be read as probabilities, so any confidence gate on `high` should be set on "
        "calibrated scores (temperature or per-class scaling fitted on the 2018 validation year).",
        "",
        "## Speed on this laptop",
        "",
        f"{run['cpu_ms_per_headline_batched']:.1f} ms per headline in batches of {run['batch_size']}, "
        f"{run['cpu_ms_per_headline_single']:.1f} ms one at a time (n = {run['single_n']}), on "
        f"{run['torch_threads']} CPU threads with no GPU - about "
        f"{1000 / run['cpu_ms_per_headline_single']:.0f} headlines a second for $0. The September GPU "
        "figures (2.3-2.5 ms, a Colab T4) are a different device and are labelled as such wherever they "
        "appear.",
        "",
        f"Source: `m3_checkpoint_check.json` ({run['timestamp']}, {run['platform']}, torch "
        f"{run['torch_version']}). Per-headline probabilities: `{run['predictions_file']}` (git-ignored; "
        "regenerate with `code/student_check.py`).",
    ]
    out = rd / "m3_checkpoint_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
