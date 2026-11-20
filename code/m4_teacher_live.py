"""M4 (b) - the teacher's real-time latency and cost, measured rather than assumed.

The labels were bought through the Batch API: half price, but asynchronous, so a batch has no
per-request latency to report, and the cost per call it implies is not what a live user would pay.
This sends a small random sample of held-out headlines through the ordinary Messages API, one at a
time, exactly as the teacher was asked (same model, effort, prompt and output schema, via
teacher_label.request_params), and records for each call its wall-clock time and its billed usage.

    python code/m4_teacher_live.py --sample data/samples/fnspid_clean_48040_seed42.csv --dry-run
    python code/m4_teacher_live.py --sample data/samples/fnspid_clean_48040_seed42.csv --n 200 --budget 1.00

--dry-run counts tokens (not billed) and prints a projected cost; nothing is sent to the model.
A real run SPENDS MONEY and stops once spending reaches --budget. Cost is computed from each
response's own usage at the published per-token rates in teacher_label.PRICES, without the batch
discount, because these are real-time calls.

The labels returned are compared with the stored teacher labels only as a sanity check that the
same task was asked; the teacher's self-consistency is measured properly elsewhere (M2).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from m1_reproduce import default_results_dir
from teacher_label import (CACHE_READ_MULT, CACHE_WRITE_MULT, PRICES, build_client, default_data_dir,
                           load_dotenv, request_params)
from train_student import HEADS, drop_near_duplicates, load_dataset, split_by_time


def call_cost(usage, in_rate: float, out_rate: float) -> float:
    """Dollars for one real-time call, from the usage the API returned."""
    write_5m = write_1h = 0
    breakdown = getattr(usage, "cache_creation", None)
    if breakdown is not None:
        write_5m = getattr(breakdown, "ephemeral_5m_input_tokens", 0) or 0
        write_1h = getattr(breakdown, "ephemeral_1h_input_tokens", 0) or 0
    else:
        write_5m = usage.cache_creation_input_tokens or 0
    return (usage.input_tokens * in_rate
            + (usage.cache_read_input_tokens or 0) * in_rate * CACHE_READ_MULT
            + write_5m * in_rate * CACHE_WRITE_MULT["5m"]
            + write_1h * in_rate * CACHE_WRITE_MULT["1h"]
            + usage.output_tokens * out_rate) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--labels", default="full")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="claude-opus-5", help="the teacher; do not change it for M4")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--budget", type=float, default=1.00, help="hard stop, in dollars")
    ap.add_argument("--dry-run", action="store_true", help="count tokens only; nothing billed")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root)
    results_dir = args.results_dir or default_results_dir()
    in_rate, out_rate = PRICES[args.model]

    # A random draw from the held-out years, so the sample is the population the student is
    # tested on rather than the first rows of a date-ordered file (the mistake behind the old
    # 93.0% ceiling).
    df = load_dataset(args.labels, args.sample, args.data_dir or default_data_dir())
    df, _ = drop_near_duplicates(df)
    _, _, test = split_by_time(df, 2018, 2019)
    sample = test.sample(n=args.n, random_state=args.seed)
    print(f"{len(sample)} headlines drawn at random (seed {args.seed}) from {len(test):,} test rows")

    client = build_client()
    if args.dry_run:
        counts = [client.messages.count_tokens(**{k: v for k, v in request_params(
            h, args.model, args.effort, args.max_tokens).items() if k != "max_tokens"}).input_tokens
            for h in sample["Article_title"].head(5)]
        per_in = float(np.mean(counts))
        print(f"input tokens per call (mean of 5): {per_in:.0f}, mostly the cached system prompt")
        print("output tokens are unknown until a call is made; the 24 Sep batch measured about 48 per "
              "item at this effort")
        cold = per_in * in_rate * CACHE_WRITE_MULT["5m"] / 1e6
        warm = (per_in * in_rate * CACHE_READ_MULT + 48 * out_rate) / 1e6
        print(f"projected: first call ${cold + 48 * out_rate / 1e6:.4f}, then about ${warm:.5f} each "
              f"-> ${cold + warm * len(sample):.2f} for {len(sample)} (budget cap ${args.budget:.2f})")
        return

    rows, spent = [], 0.0
    for i, r in enumerate(sample.itertuples(index=False), start=1):
        params = request_params(r.Article_title, args.model, args.effort, args.max_tokens)
        t0 = time.perf_counter()
        msg = client.messages.create(**params)
        ms = (time.perf_counter() - t0) * 1000
        cost = call_cost(msg.usage, in_rate, out_rate)
        spent += cost
        text = next((b.text for b in msg.content if b.type == "text"), None)
        try:
            label = json.loads(text) if text else {}
        except json.JSONDecodeError:
            label = {}
        rows.append({"ms": ms, "cost": cost, "stop_reason": msg.stop_reason,
                     "input": msg.usage.input_tokens, "output": msg.usage.output_tokens,
                     "cache_read": msg.usage.cache_read_input_tokens or 0,
                     "cache_write": msg.usage.cache_creation_input_tokens or 0,
                     "agree": {h: label.get(h) == getattr(r, h) for h in HEADS},
                     "parsed": bool(label)})
        if i % 25 == 0 or i == len(sample):
            print(f"  {i}/{len(sample)}  median {np.median([x['ms'] for x in rows]):.0f} ms  "
                  f"spent ${spent:.4f}", flush=True)
        if spent >= args.budget:
            print(f"budget ${args.budget:.2f} reached after {i} calls - stopping")
            break

    ms = np.array([x["ms"] for x in rows])
    warm = [x for x in rows[1:]]  # the first call writes the cache; the rest are steady state
    per_1000_warm = float(np.mean([x["cost"] for x in warm]) * 1000) if warm else None
    run = {
        "milestone": "M4", "arm": "teacher, real-time Messages API",
        "model": args.model, "effort": args.effort, "n_calls": len(rows),
        "sample": f"random {args.n} of the {len(test):,} test headlines, seed {args.seed}",
        "latency_ms": {"median": float(np.median(ms)), "mean": float(ms.mean()),
                       "p10": float(np.percentile(ms, 10)), "p90": float(np.percentile(ms, 90)),
                       "max": float(ms.max())},
        "cost_total_usd": spent,
        "cost_per_1000_usd_warm_cache": per_1000_warm,
        "first_call_cost_usd": rows[0]["cost"],
        "tokens": {k: int(sum(x[k] for x in rows)) for k in ("input", "output", "cache_read", "cache_write")},
        "output_tokens_per_call_mean": float(np.mean([x["output"] for x in rows])),
        "parsed": int(sum(x["parsed"] for x in rows)),
        "stop_reasons": {s: sum(x["stop_reason"] == s for x in rows) for s in {x["stop_reason"] for x in rows}},
        "agreement_with_stored_teacher_labels": {h: float(np.mean([x["agree"][h] for x in rows])) for h in HEADS},
        "rates_per_mtok": {"input": in_rate, "output": out_rate, "cache_read_mult": CACHE_READ_MULT,
                           "cache_write_5m_mult": CACHE_WRITE_MULT["5m"], "batch_discount": "not applied"},
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.machine()} python {platform.python_version()}",
    }
    out = results_dir / "m4_teacher_live.json"
    out.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nmedian {run['latency_ms']['median']:.0f} ms, p90 {run['latency_ms']['p90']:.0f} ms; "
          f"${per_1000_warm:.3f} per 1,000 warm; total spent ${spent:.4f}\nwrote {out}")


if __name__ == "__main__":
    main()
