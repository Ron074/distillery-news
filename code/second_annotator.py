"""Prepare the gold headlines for a second annotator, and read its answers back.

The teacher cannot check itself and Claude cannot check the teacher - same model, so the result
would be the teacher grading its own work. A different model family is not a human, but it is
genuinely independent weights, and it answers the narrower question the gold sheet exists for:
is the taxonomy written clearly enough that an independent reader lands where the teacher lands?

    python code/second_annotator.py prompt                      # writes chunk_01.txt ...
    # paste each chunk into the other model, save its reply as reply_01.txt ...
    python code/second_annotator.py parse --model gpt-5
    python code/score_gold.py --gold data/gold/second_annotator/gold_gpt-5.csv --tag full

Report what this is: **cross-model agreement**, not human or expert validation. Two frontier
models share most of their training corpus and much of their instruction tuning, so their errors
are correlated in ways a human's would not be. Agreement here says the task is model-consensual.

The rubric comes from `taxonomy.SYSTEM_PROMPT` - the exact bytes the teacher was given - rather
than from a restatement, because a reworded rubric measures the difference between two prompts
instead of the difference between two models.

Two honest limitations, both stated rather than hidden:

  1. The teacher saw one headline per request with no other context. Pasting 25 at a time lets
     the second annotator calibrate across the batch - noticing, say, that many are roundups -
     which the teacher could not do. The instruction tells it to judge each independently; that
     mitigates and does not eliminate. Smaller --chunk-size narrows the gap at the cost of more
     pasting.
  2. The teacher saw the headline ALONE: no ticker, no date. So do these chunks. Adding the
     ticker would hand the second annotator information the teacher never had.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

import taxonomy
from teacher_label import default_data_dir

UNSURE = "unsure"
ALLOWED = {"category": taxonomy.CATEGORIES,
           "materiality": taxonomy.MATERIALITY,
           "direction": taxonomy.DIRECTION}

# Tolerant on how the model punctuates, strict on what the fields may contain. Accepts
# "1. earnings | medium", "1, earnings, medium", "1) earnings medium".
REPLY_LINE = re.compile(r"^\s*(\d+)\s*[.):,]?\s*([A-Za-z]+)\s*[|,;]?\s*([A-Za-z]*)\s*$")

HEADER = """You are labelling financial news headlines for a research dataset.

Use ONLY the rubric below. Do not use outside knowledge about any company, and do not let the
other headlines in this list influence how you read any single one - judge each on its own, as if
it were the only headline you had been given.

================================ RUBRIC ================================
{rubric}
========================================================================

Label these {n} headlines:

{headlines}

Output exactly {n} lines and nothing else - no preamble, no commentary, no blank lines between:

{example}

Every number from 1 to {n} must appear exactly once, in order. Use lowercase. If the rubric
genuinely does not resolve a field for some headline, write `unsure` for that field.
"""


def chunk_dir(data_dir: Path, name: str) -> Path:
    return data_dir / "gold" / name


def mode_prompt(args) -> None:
    data_dir = args.data_dir or default_data_dir()
    out_dir = chunk_dir(data_dir, args.out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.gold, dtype=str)
    # The sheet's own label columns are ignored on purpose: this reads the headlines only, so
    # nothing the teacher or anyone else has written can reach the second annotator.
    for column in ("uid", "Article_title"):
        if column not in df.columns:
            raise SystemExit(f"{args.gold} has no {column} column")
    df = df.dropna(subset=["Article_title"]).reset_index(drop=True)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    # Placeholders rather than sample labels: any concrete example pulls the answers toward the
    # classes it happens to name. If the model echoes a placeholder instead of choosing, the
    # parser rejects it by name - a loud failure, where a biased example would have been silent.
    example = "\n".join(f"{i}. {' | '.join('<' + f + '>' for f in fields)}" for i in (1, 2))

    # Records WHICH rubric these answers were given under. If taxonomy.py is edited later, the
    # hash no longer matches and the comparison is known to be stale rather than assumed current.
    rubric_sha = hashlib.sha256(taxonomy.SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
    manifest = {"gold_sheet": str(args.gold), "fields": fields, "chunk_size": args.chunk_size,
                "rubric_sha": rubric_sha, "chunks": {}}
    for index, start in enumerate(range(0, len(df), args.chunk_size), start=1):
        part = df.iloc[start:start + args.chunk_size]
        # Numbered 1..n within the chunk rather than echoing uids: a model transcribing a
        # 17-character hash will occasionally corrupt one, and a corrupted uid is a row that
        # silently vanishes from the comparison. A small integer it can copy reliably, and the
        # parser can then demand that every one of them came back.
        headlines = "\n".join(f"{i}. {str(t).strip()}" for i, t in enumerate(part["Article_title"], start=1))
        text = HEADER.format(rubric=taxonomy.SYSTEM_PROMPT.strip(), n=len(part),
                             headlines=headlines, example=example)
        path = out_dir / f"chunk_{index:02d}.txt"
        path.write_text(text, encoding="utf-8")
        manifest["chunks"][f"{index:02d}"] = list(part["uid"])

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    n_chunks = len(manifest["chunks"])
    print(f"wrote {n_chunks} chunk file(s) and manifest.json to {out_dir}")
    print(f"  {len(df)} headlines, {args.chunk_size} per chunk, fields: {', '.join(fields)}")
    print("\nFor each chunk: paste its whole contents into the other model, then save the reply")
    print(f"verbatim as {out_dir / 'reply_01.txt'} (reply_02.txt, ...).")
    print("\nDo NOT show it the teacher's labels, and do not ask it whether it agrees with a label.")
    print("Shown a proposed answer, a model defers to it - that measures deference, not judgement.")


def parse_reply(text: str, uids: list[str], fields: list[str], chunk: str) -> dict[str, dict]:
    """Read one reply. Refuses on anything missing, duplicated or out of vocabulary."""
    seen: dict[int, dict] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        m = REPLY_LINE.match(line)
        if not m:
            continue  # prose the model added around its answer; the completeness check below is
                      # what guarantees nothing real was skipped
        number = int(m.group(1))
        values = [m.group(2).lower(), m.group(3).lower()]
        row = {}
        for position, field in enumerate(fields):
            value = values[position] if position < len(values) else ""
            if value not in set(ALLOWED[field]) | {UNSURE}:
                raise SystemExit(
                    f"reply_{chunk}.txt line {number}: {field} is '{value}', which is not in the "
                    f"vocabulary.\n  allowed: {', '.join(ALLOWED[field])} (or '{UNSURE}')\n"
                    "  Fix that line in the reply file and run parse again."
                )
            row[field] = value
        if number in seen:
            raise SystemExit(f"reply_{chunk}.txt gives line {number} more than once; "
                             "remove the duplicate and run parse again")
        seen[number] = row

    # Completeness is checked rather than tolerated. A reply that quietly skipped three headlines
    # would otherwise shrink the comparison and leave the agreement on the rest looking better,
    # which is the exact shape of every silent bug this project has had to undo.
    missing = [i for i in range(1, len(uids) + 1) if i not in seen]
    extra = [i for i in seen if i > len(uids)]
    if missing or extra:
        raise SystemExit(
            f"reply_{chunk}.txt does not line up with chunk_{chunk}.txt ({len(uids)} headlines):\n"
            + (f"  missing numbers: {missing}\n" if missing else "")
            + (f"  numbers beyond the chunk: {extra}\n" if extra else "")
            + "  Ask the model for the missing lines, or re-paste that chunk, then run parse again."
        )
    return {uids[number - 1]: row for number, row in seen.items()}


def mode_parse(args) -> None:
    data_dir = args.data_dir or default_data_dir()
    out_dir = chunk_dir(data_dir, args.out_name)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    fields = manifest["fields"]

    labels: dict[str, dict] = {}
    missing_files = []
    for chunk, uids in manifest["chunks"].items():
        path = out_dir / f"reply_{chunk}.txt"
        if not path.exists():
            missing_files.append(path.name)
            continue
        labels.update(parse_reply(path.read_text(encoding="utf-8"), uids, fields, chunk))

    if missing_files:
        print(f"no reply yet for: {', '.join(missing_files)}")
        print("  (parsing the rest; re-run once they are saved)")
    if not labels:
        raise SystemExit("no replies found - save the model's answers as reply_01.txt and so on")

    gold = pd.read_csv(manifest["gold_sheet"], dtype=str)
    for field in fields:
        gold[field] = gold["uid"].map(lambda u: labels.get(u, {}).get(field, ""))

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    out = out_dir / f"gold_{slug}.csv"
    gold.to_csv(out, index=False)

    meta = {"annotator": args.model, "fields": fields, "n_labelled": len(labels),
            "n_in_sheet": len(gold), "chunk_size": manifest["chunk_size"],
            "gold_sheet": manifest["gold_sheet"],
            "note": "cross-model agreement, not human validation; same rubric bytes as the "
                    "teacher's system prompt; labelled blind, never shown the teacher's labels"}
    (out_dir / f"meta_{slug}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(labels)} of {len(gold)} rows labelled by {args.model}")
    for field in fields:
        filled = gold[gold[field] != ""]
        counts = {k: int(v) for k, v in filled[field].value_counts().items()}
        print(f"\n  {field}: {counts}")
        if (filled[field] == UNSURE).any():
            print(f"    ({int((filled[field] == UNSURE).sum())} marked unsure)")
    print(f"\nwrote {out}")
    print(f"      {out_dir / f'meta_{slug}.json'}")
    print("\nScore it the same way a hand-labelled sheet is scored:")
    print(f"  python code/score_gold.py --gold {out} --tag full --fields {','.join(fields)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("prompt", help="write the chunk files to paste into the other model")
    p1.add_argument("--gold", type=Path, default=Path("data/gold/gold_blank_pilot.csv"))
    p1.add_argument("--chunk-size", type=int, default=25)
    p1.add_argument("--fields", default="category,materiality")

    p2 = sub.add_parser("parse", help="read the replies back into a scoreable sheet")
    p2.add_argument("--model", required=True,
                    help="which model answered, recorded alongside the labels; these change "
                         "under you, so 'gpt-5' beats 'chatgpt'")

    for p in (p1, p2):
        p.add_argument("--data-dir", type=Path, default=None)
        p.add_argument("--out-name", default="second_annotator")

    args = ap.parse_args()
    {"prompt": mode_prompt, "parse": mode_parse}[args.mode](args)


if __name__ == "__main__":
    main()
