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

# Published per-MTok rates. Batch is half price; cache reads are a tenth of the input rate.
# Verify against anthropic.com/pricing before quoting a number in the report.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
BATCH_DISCOUNT = 0.5
CACHE_READ_MULT = 0.1
# A cache write is billed by its lifetime: 1.25x input for the 5-minute default, 2x for 1h.
CACHE_WRITE_MULT = {"5m": 1.25, "1h": 2.0}


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


def request_params(headline: str, model: str, effort: str, max_tokens: int,
                   cache_ttl: str = "") -> dict:
    # Empty ttl means the 5-minute default, which is what batch traffic wants: requests sharing a
    # prefix start close together and keep the entry warm on their own. A measured run on "1h"
    # cost 6x the best 5m run -- the longer lifetime bought nothing and doubled the write price.
    cache_control = {"type": "ephemeral"}
    if cache_ttl:
        cache_control["ttl"] = cache_ttl
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{
            "type": "text",
            "text": taxonomy.SYSTEM_PROMPT,
            "cache_control": cache_control,
        }],
        "messages": [{"role": "user", "content": headline}],
        "output_config": {"format": taxonomy.OUTPUT_CONFIG_FORMAT, "effort": effort},
    }


def load_sample(path: Path, limit: int | None, offset: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    missing = {"uid", "Article_title"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    df = df.iloc[offset:]
    return df.head(limit) if limit else df


DUP_SUFFIX = "_dup"


def mode_export_gold(args) -> None:
    """Write the pilot headlines with blank label columns, for hand-labeling before seeing the teacher."""
    data_dir = args.data_dir or default_data_dir()
    # Load the whole pool before narrowing: truncating first and shuffling after would just
    # reorder an already-biased slice.
    df = load_sample(args.sample, None if args.shuffle else args.limit, args.offset)

    if args.labeled_tag:
        # A gold row the teacher never labeled cannot be compared against anything.
        labeled = done_uids(labels_path(data_dir, args.labeled_tag))
        if not labeled:
            raise SystemExit(f"no labels found for tag '{args.labeled_tag}'")
        df = df[df["uid"].isin(labeled)]
        print(f"drawing from {len(df):,} rows labeled under tag '{args.labeled_tag}'")

    if args.shuffle:
        # The sample file may be in a meaningful order (an early version sorted by date), which
        # would make the head of it a biased slice rather than a representative one.
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True).head(args.limit)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    out = df[["uid", "Date", "Stock_symbol", "Article_title"]].copy()
    for column in fields:
        out[column] = ""

    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    csv_path = gold_dir / f"gold_blank_{args.tag}.csv"
    out.to_csv(csv_path, index=False)

    allowed = {"category": taxonomy.CATEGORIES, "materiality": taxonomy.MATERIALITY,
               "direction": taxonomy.DIRECTION}
    key_path = gold_dir / "labeling_key.txt"
    key_path.write_text(
        f"Fill {' / '.join(fields)} for each row, then save.\n"
        "Do this BEFORE looking at any teacher labels - the comparison is only\n"
        "meaningful if your judgement was formed independently.\n"
        "Unsure on a row? Write 'unsure' rather than guessing; those are excluded,\n"
        "and which rows are hard is itself a finding worth reporting.\n\n"
        + "".join(f"{f:12s}: {', '.join(allowed[f])}\n" for f in fields if f in allowed)
        + "\n" + taxonomy.SYSTEM_PROMPT,
        encoding="utf-8",
    )
    print(f"wrote {csv_path}  ({len(out):,} rows to label)")
    print(f"wrote {key_path}  (the definitions, so you need not switch windows)")


def report_consistency(path: Path) -> None:
    """Compare any headline the teacher was asked about twice."""
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_uid = {r["uid"]: r for r in rows}
    pairs = [(by_uid[u[: -len(DUP_SUFFIX)]], by_uid[u])
             for u in by_uid if u.endswith(DUP_SUFFIX) and u[: -len(DUP_SUFFIX)] in by_uid]
    if not pairs:
        return

    fields = ("category", "materiality", "direction")
    agree = {f: sum(a[f] == b[f] for a, b in pairs) for f in fields}
    all_three = sum(all(a[f] == b[f] for f in fields) for a, b in pairs)
    print(f"\nteacher self-consistency over {len(pairs)} repeated headlines:")
    for f in fields:
        print(f"  {f:12s} {agree[f] / len(pairs):6.1%}")
    print(f"  {'all three':12s} {all_three / len(pairs):6.1%}   <- ceiling on student fidelity")


def labels_path(data_dir: Path, tag: str) -> Path:
    return data_dir / "labels" / f"teacher_{tag}.jsonl"


def batch_state_path(data_dir: Path, tag: str) -> Path:
    return data_dir / "labels" / f"teacher_{tag}.batch.json"


def done_uids(path: Path) -> set[str]:
    """Uids already saved. Tolerates a truncated final line: a connection dropped mid-write leaves
    one, and refusing to parse it would strand every label already paid for behind a crash."""
    if not path.exists():
        return set()
    uids = set()
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                uids.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                print(f"  skipping unreadable line {n} in {path.name} (it will be re-requested)")
    return uids


# ------------------------------------------------------------------------------------- modes

def mode_estimate(args) -> None:
    client = build_client()
    df = load_sample(args.sample, args.limit, args.offset)
    probe = df["Article_title"].head(args.probe).tolist()

    counted = []
    for headline in probe:
        params = request_params(headline, args.model, args.effort, args.max_tokens, args.cache_ttl)
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
    df = load_sample(args.sample, args.limit, args.offset)

    already = done_uids(labels_path(data_dir, args.tag))
    todo = df[~df["uid"].isin(already)]
    if already:
        print(f"{len(already):,} uids already labeled; {len(todo):,} remaining")
    if todo.empty:
        print("nothing to do")
        return
    if args.duplicates:
        # Opus 5 has no temperature control, so run-to-run variation cannot be switched off.
        # Asking about the same headlines twice measures it, and that is the ceiling on fidelity.
        dup = todo.head(args.duplicates).copy()
        dup["uid"] = dup["uid"] + DUP_SUFFIX
        todo = pd.concat([todo, dup], ignore_index=True)
        print(f"including {len(dup)} repeated headlines to measure teacher self-consistency")

    if len(todo) > 100_000:
        raise SystemExit("a single batch takes at most 100,000 requests; split the sample")

    requests = [
        Request(
            custom_id=row.uid,
            params=MessageCreateParamsNonStreaming(
                **request_params(row.Article_title, args.model, args.effort, args.max_tokens, args.cache_ttl)
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
        "cache_ttl": args.cache_ttl,
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

    import anthropic

    # The batch runs server-side, so losing the network is a pause, not a failure. Sleeping
    # through a disconnection lets an unattended collector finish on its own once a laptop is
    # back online, instead of dying minutes after it is unplugged.
    offline_for = 0
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            offline_for += args.poll_seconds
            print(f"no connection; retrying in {args.poll_seconds}s "
                  f"(offline {offline_for // 60}m so far)", flush=True)
            time.sleep(args.poll_seconds)
            continue
        offline_for = 0
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

    usage = {"input": 0, "cache_read": 0, "write_5m": 0, "write_1h": 0, "output": 0}
    written = errored = skipped = 0
    interrupted = None
    try:
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
                # Writes are priced by lifetime, so split them rather than assuming one rate.
                breakdown = getattr(msg.usage, "cache_creation", None)
                if breakdown is not None:
                    usage["write_5m"] += getattr(breakdown, "ephemeral_5m_input_tokens", 0) or 0
                    usage["write_1h"] += getattr(breakdown, "ephemeral_1h_input_tokens", 0) or 0
                else:
                    key = "write_1h" if state.get("cache_ttl") == "1h" else "write_5m"
                    usage[key] += msg.usage.cache_creation_input_tokens or 0
                fh.write(json.dumps({"uid": result.custom_id, **label,
                                     "model": state["model"], "effort": state["effort"]}) + "\n")
                fh.flush()
                written += 1
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
        # Every label written so far is on disk; re-running resumes from there.
        interrupted = exc

    in_rate, out_rate = PRICES.get(state["model"], PRICES["claude-opus-5"])
    cost = (
        usage["input"] * in_rate
        + usage["cache_read"] * in_rate * CACHE_READ_MULT
        + usage["write_5m"] * in_rate * CACHE_WRITE_MULT["5m"]
        + usage["write_1h"] * in_rate * CACHE_WRITE_MULT["1h"]
        + usage["output"] * out_rate
    ) / 1e6 * BATCH_DISCOUNT

    print(f"\nwrote {written:,} labels -> {out_path}")
    if interrupted is not None:
        print(f"*** connection lost mid-download ({type(interrupted).__name__}) ***")
        print(f"    {written:,} labels are saved; re-run this same command to fetch the rest")
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
    report_consistency(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--model", default="claude-opus-5")
        p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--max-tokens", type=int, default=2048)
        p.add_argument("--data-dir", type=Path, default=None)
        p.add_argument("--cache-ttl", default="",
                       help="cache lifetime for the shared taxonomy prefix; empty string for the 5m default")
        p.add_argument("--offset", type=int, default=0,
                       help="skip this many rows first, to draw a slice not already labeled")

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
    p_sub.add_argument("--duplicates", type=int, default=0,
                       help="also ask about this many headlines twice, to measure teacher self-consistency")
    common(p_sub)

    p_gold = sub.add_parser("export-gold", help="write blank label sheets for hand-labeling; spends nothing")
    p_gold.add_argument("--sample", type=Path, required=True)
    p_gold.add_argument("--limit", type=int, default=100)
    p_gold.add_argument("--tag", default="pilot")
    p_gold.add_argument("--labeled-tag", default=None,
                        help="restrict to rows already labeled under this run tag, so they compare")
    p_gold.add_argument("--shuffle", action="store_true",
                        help="draw the rows at random rather than taking the head of the file")
    p_gold.add_argument("--seed", type=int, default=42)
    p_gold.add_argument("--fields", default="category,materiality",
                        help="comma-separated label columns to leave blank for hand-labeling")
    common(p_gold)

    p_col = sub.add_parser("collect", help="poll a submitted batch and write its labels")
    p_col.add_argument("--tag", required=True)
    p_col.add_argument("--wait", action="store_true")
    p_col.add_argument("--poll-seconds", type=int, default=60)
    common(p_col)

    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent)
    {"estimate": mode_estimate, "submit": mode_submit, "collect": mode_collect,
     "export-gold": mode_export_gold}[args.mode](args)


if __name__ == "__main__":
    main()
