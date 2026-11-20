"""M6 step 1 - the student's raw logits on the 2018 validation year and the 2019+ test split.

    python code/m6_logits.py --checkpoint <models dir>/stat_finbert_3head_<date> \
        --sample data/samples/fnspid_clean_48040_seed42.csv

Calibration is fitted on validation and judged on test, so it needs both. It also needs logits
rather than the saved probabilities: those are rounded to six decimals, and the log of a rounded-
to-zero probability is minus infinity. Writes results/logits_val.npz and results/logits_test.npz
(git-ignored: regenerable, and only useful beside the checkpoint that produced them).

The test logits must reproduce the M3 accuracies exactly; the script stops if they do not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from m1_reproduce import default_results_dir
from student_check import load_student
from teacher_label import default_data_dir
from train_student import drop_near_duplicates, encode_batch, load_dataset, split_by_time


@torch.no_grad()
def logits_for(model, tokenizer, texts, heads, batch_size, max_length) -> dict[str, np.ndarray]:
    model.eval()
    out = {h: [] for h in heads}
    for i in range(0, len(texts), batch_size):
        z = model(**encode_batch(tokenizer, texts[i:i + batch_size], max_length, "cpu"))
        for h in heads:
            out[h].append(z[h].float().cpu().numpy())
    return {h: np.concatenate(v) for h, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--labels", default="full")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    rd = args.results_dir or default_results_dir()

    model, tokenizer, heads, meta = load_student(args.checkpoint)
    df = load_dataset(args.labels, args.sample, args.data_dir or default_data_dir())
    df, _ = drop_near_duplicates(df)
    _, val, test = split_by_time(df, meta["split"]["val_year"], meta["split"]["test_from"])
    if (len(val), len(test)) != (meta["n_val"], meta["n_test"]):
        raise SystemExit(f"split sizes {len(val)}/{len(test)} != checkpoint's {meta['n_val']}/{meta['n_test']}")

    m3 = json.loads((rd / "m3_3head.json").read_text(encoding="utf-8"))
    for name, frame in (("val", val), ("test", test)):
        z = logits_for(model, tokenizer, frame["Article_title"].tolist(), heads,
                       args.batch_size, meta["max_length"])
        teacher = {h: np.array([heads[h].index(v) for v in frame[h]]) for h in heads}
        for h in heads:
            acc = float((z[h].argmax(1) == teacher[h]).mean())
            print(f"  {name:4s} {h:12s} acc {acc:.4f}", flush=True)
            if name == "test" and acc != m3["test"][h]["accuracy"]:
                raise SystemExit(f"test {h} accuracy {acc} != M3 {m3['test'][h]['accuracy']}")
        np.savez_compressed(
            rd / f"logits_{name}.npz", uid=frame["uid"].to_numpy(dtype=str),
            **{f"z_{h}": z[h] for h in heads}, **{f"y_{h}": teacher[h] for h in heads},
            label_sets=json.dumps(heads), checkpoint_sha256=meta["sha256"])
        print(f"wrote {rd / f'logits_{name}.npz'} ({len(frame):,} rows)", flush=True)


if __name__ == "__main__":
    main()
