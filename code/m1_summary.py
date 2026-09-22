"""Collect the M1 run JSONs into one comparison table.

    python code/m1_summary.py            # -> results/m1_summary.md

Regenerate this rather than hand-editing it, so the table can never drift from the runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from m1_reproduce import default_results_dir

ARM_ORDER = {"baseline": 0, "finetune": 1, "zeroshot": 2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()

    results_dir = args.results_dir or default_results_dir()
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(results_dir.glob("m1_*.json"))]
    if not runs:
        raise SystemExit(f"no m1_*.json runs found in {results_dir}")
    runs.sort(key=lambda r: (ARM_ORDER.get(r["arm"], 9), -r["metrics"]["macro_f1"]))

    first = runs[0]
    lines = [
        "# M1 — FinBERT reproduction on Financial PhraseBank",
        "",
        f"Corpus: Financial PhraseBank `{first['agreement']}` (Malo et al. 2014). "
        f"Deterministic stratified split, seed {first['seed']}: "
        f"{first['n_train']} train / {first['n_val']} val / {first['n_test']} test.",
        f"All arms scored on the same held-out test set. Hardware: {first['platform']}, CPU.",
        "",
        "| arm | model | accuracy | macro F1 | weighted F1 | train | inference (ms/sentence) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        m = r["metrics"]
        train = f"{r['train_seconds'] / 60:.1f} min" if r.get("train_seconds") else "—"
        ms = r["infer_seconds"] / r["n_test"] * 1000 if r.get("infer_seconds") else None
        lines.append(
            f"| `{r['arm']}` | {r['model']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['weighted_f1']:.4f} | {train} | {ms:.1f} |" if ms is not None else
            f"| `{r['arm']}` | {r['model']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['weighted_f1']:.4f} | {train} | — |"
        )

    notes = [r for r in runs if r.get("notes")]
    if notes:
        lines += ["", "## Notes", ""]
        for r in notes:
            lines.append(f"- **{r['model']}** — {r['notes']}")

    lines += ["", "## Per-class F1", "", "| model | negative | neutral | positive |", "|---|---|---|---|"]
    for r in runs:
        pc = r["metrics"]["per_class"]
        lines.append(f"| {r['model']} | " + " | ".join(f"{pc[c]['f1-score']:.4f}" for c in ("negative", "neutral", "positive")) + " |")

    out = results_dir / "m1_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
