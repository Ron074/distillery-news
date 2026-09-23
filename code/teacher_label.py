"""M2 - label an FNSPID sample with the LLM teacher, via the Batch API.

Three modes, in the order you should use them:

    python code/teacher_label.py estimate --sample data/samples/fnspid_sample_50000_seed42.csv
    python code/teacher_label.py submit   --sample ... --limit 200 --tag pilot
    python code/teacher_label.py collect  --tag pilot

`estimate` spends nothing: token counting is not billed. It reports exact input cost and brackets
the output cost, which cannot be known before a real run because thinking tokens dominate it.
`submit` creates a batch and records its id. `collect` polls, writes labels, and reports the cost
that was actually incurred.

Labels are written as JSONL keyed by `uid`; re-running skips any uid already present, so an
interrupted job never re-pays for work it already has.

The API key is read from the environment ($ANTHROPIC_API_KEY) or an external .env - never committed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

import taxonomy

# Published per-MTok rates. Batch is half price; cache reads are a tenth of the input rate and
# cache writes 1.25x. Verify against anthropic.com/pricing before quoting a number in the report.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
BATCH_DISCOUNT = 0.5
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25


def default_data_dir() -> Path:
    env = os.environ.get("DISTILLERY_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parent.parent / "data"


def load_dotenv(repo_root: Path) -> None:
    """Read KEY=value lines from an external .env if the variable is not already set."""
    env_path = Path(os.environ.get("DISTILLERY_ENV", repo_root / ".env"))
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_client():
    import anthropic

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit(
            "No API credentials found. Set ANTHROPIC_API_KEY in your environment or put it in a\n"
            ".env file at the repo root (git-ignored), then re-run."
        )
    return anthropic.Anthropic()


def request_params(headline: str, model: str, effort: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{
            "type": "text",
            "text": taxonomy.SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{"role": "user", "content": headline}],
        "output_config": {"format": taxonomy.OUTPUT_CONFIG_FORMAT, "effort": effort},
    }


def load_sample(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    missing = {"uid", "Article_title"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    return df.head(limit) if limit else df


def labels_path(data_dir: Path, tag: str) -> Path:
    return data_dir / "labels" / f"teacher_{tag}.jsonl"


def batch_state_path(data_dir: Path, tag: str) -> Path:
    return data_dir / "labels" / f"teacher_{tag}.batch.json"


def done_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    uids = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                uids.add(json.loads(line)["uid"])
    return uids


# ------------------------------------------------------------------------------------- modes

def mode_estimate(args) -> None:
    client = build_client()
    df = load_sample(args.sample, args.limit)
    probe = df["Article_title"].head(args.probe).tolist()

    counted = []
    for headline in probe:
        params = request_params(headline, args.model, args.effort, args.max_tokens)
        resp = client.messages.count_tokens(
            model=args.model, system=params["system"], messages=params["messages"]
        )
        counted.append(resp.input_tokens)
    mean_input = sum(counted) / len(counted)

    in_rate, out_rate = PRICES.get(args.model, PRICES["claude-opus-5"])
    n = len(df)
    # The cached taxonomy prefix is written once and read on every later request.
    system_tokens = min(counted)  # dominated by the shared prefix
    fresh = max(mean_input - system_tokens, 0.0)

    cache_write = system_tokens * in_rate * CACHE_WRITE_MULT / 1e6
    per_item_in = (system_tokens * in_rate * CACHE_READ_MULT + fresh * in_rate) / 1e6

    print(f"sample          : {args.sample.name}")
    print(f"items           : {n:,}")
    print(f"model / effort  : {args.model} / {args.effort}")
    print(f"mean input tok  : {mean_input:.0f}  (cached prefix ~{system_tokens:.0f}, fresh ~{fresh:.0f})")
    print(f"\ninput cost      : ${(per_item_in * n + cache_write) * BATCH_DISCOUNT:,.2f} (batch price)")
    print("\noutput cost depends on how many thinking tokens the model spends per item,")
    print("which is not knowable before a real run. Brackets at batch pricing:\n")
    print(f"  {'out tok/item':>13} {'output $':>10} {'TOTAL $':>10}")
    for out_tok in (100, 200, 400, 800):
        out_cost = out_tok * out_rate / 1e6 * n * BATCH_DISCOUNT
        total = out_cost + (per_item_in * n + cache_write) * BATCH_DISCOUNT
        print(f"  {out_tok:>13} {out_cost:>10,.2f} {total:>10,.2f}")
    print("\nRun a pilot to replace these brackets with a measured number:")
    print(f"  python code/teacher_label.py submit --sample {args.sample} --limit 200 --tag pilot")


def mode_submit(args) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = build_client()
    data_dir = args.data_dir or default_data_dir()
    df = load_sample(args.sample, args.limit)

    already = done_uids(labels_path(data_dir, args.tag))
    todo = df[~df["uid"].isin(already)]
    if already:
        print(f"{len(already):,} uids already labeled; {len(todo):,} remaining")
    if todo.empty:
        print("nothing to do")
        return
    if len(todo) > 100_000:
        raise SystemExit("a single batch takes at most 100,000 requests; split the sample")

    requests = [
        Request(
            custom_id=row.uid,
            params=MessageCreateParamsNonStreaming(
                **request_params(row.Article_title, args.model, args.effort, args.max_tokens)
            ),
        )
        for row in todo.itertuples()
    ]

    print(f"submitting {len(requests):,} requests on {args.model} (effort={args.effort})")
    if not args.yes:
        confirm = input("this spends real money. type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            raise SystemExit("aborted")

    batch = client.messages.batches.create(requests=requests)
    state_path = batch_state_path(data_dir, args.tag)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "batch_id": batch.id, "tag": args.tag, "model": args.model, "effort": args.effort,
        "n_requests": len(requests), "sample": str(args.sample),
        "created_at": str(batch.created_at),
    }, indent=2), encoding="utf-8")

    print(f"batch id : {batch.id}  ({batch.processing_status})")
    print(f"state    : {state_path}")
    print(f"\ncollect when it finishes:\n  python code/teacher_label.py collect --tag {args.tag}")


def mode_collect(args) -> None:
    client = build_client()
    data_dir = args.data_dir or default_data_dir()
    state_path = batch_state_path(data_dir, args.tag)
    if not state_path.exists():
        raise SystemExit(f"no batch state at {state_path}; submit first")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch_id = state["batch_id"]

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"{batch.processing_status}: processing={counts.processing} "
              f"succeeded={counts.succeeded} errored={counts.errored}", flush=True)
        if not args.wait:
            print("still running; re-run with --wait to block until it finishes")
            return
        time.sleep(args.poll_seconds)

    out_path = labels_path(data_dir, args.tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = done_uids(out_path)

    usage = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    written = errored = skipped = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for result in client.messages.batches.results(batch_id):
            if result.custom_id in already:
                skipped += 1
                continue
            if result.result.type != "succeeded":
                errored += 1
                continue
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), None)
            if text is None:
                errored += 1
                continue
            try:
                label = json.loads(text)
            except json.JSONDecodeError:
                errored += 1
                continue
            usage["input"] += msg.usage.input_tokens
            usage["output"] += msg.usage.output_tokens
            usage["cache_read"] += msg.usage.cache_read_input_tokens or 0
            usage["cache_write"] += msg.usage.cache_creation_input_tokens or 0
            fh.write(json.dumps({"uid": result.custom_id, **label,
                                 "model": state["model"], "effort": state["effort"]}) + "\n")
            written += 1

    in_rate, out_rate = PRICES.get(state["model"], PRICES["claude-opus-5"])
    cost = (
        usage["input"] * in_rate
        + usage["cache_read"] * in_rate * CACHE_READ_MULT
        + usage["cache_write"] * in_rate * CACHE_WRITE_MULT
        + usage["output"] * out_rate
    ) / 1e6 * BATCH_DISCOUNT

    print(f"\nwrote {written:,} labels -> {out_path}")
    if skipped:
        print(f"skipped {skipped:,} already present")
    if errored:
        print(f"*** {errored:,} requests failed or returned unparseable output ***")
    print("\nmeasured usage:")
    for k, v in usage.items():
        print(f"  {k:12s} {v:>12,} tok")
    print(f"\nMEASURED COST : ${cost:,.4f} for {written:,} items")
    if written:
        print(f"per item      : ${cost / written:.6f}")
        for n in (20_000, 50_000):
            print(f"extrapolated to {n:,}: ${cost / written * n:,.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--model", default="claude-opus-5")
        p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--max-tokens", type=int, default=2048)
        p.add_argument("--data-dir", type=Path, default=None)

    p_est = sub.add_parser("estimate", help="count tokens and bracket the cost; spends nothing")
    p_est.add_argument("--sample", type=Path, required=True)
    p_est.add_argument("--limit", type=int, default=None)
    p_est.add_argument("--probe", type=int, default=25, help="headlines to token-count")
    common(p_est)

    p_sub = sub.add_parser("submit", help="create a batch; this spends money")
    p_sub.add_argument("--sample", type=Path, required=True)
    p_sub.add_argument("--limit", type=int, default=None)
    p_sub.add_argument("--tag", required=True)
    p_sub.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    common(p_sub)

    p_col = sub.add_parser("collect", help="poll a submitted batch and write its labels")
    p_col.add_argument("--tag", required=True)
    p_col.add_argument("--wait", action="store_true")
    p_col.add_argument("--poll-seconds", type=int, default=60)
    common(p_col)

    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent)
    {"estimate": mode_estimate, "submit": mode_submit, "collect": mode_collect}[args.mode](args)


if __name__ == "__main__":
    main()
