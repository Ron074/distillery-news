"""One headline, one keystroke per field. Fills the gold sheet without a spreadsheet.

    python code/label_gold.py

Shows the headline ALONE - no ticker, no date - because that is all the teacher was given. Showing
more would mean judging a richer input than the teacher judged, and the disagreements that caused
would look like disagreements about the taxonomy when they were really about the extra context.

Every answer is written to the sheet immediately, so closing the terminal loses at most the row in
progress. Re-running resumes: rows that already have values are skipped unless --redo.

The teacher's labels are never read by this script. It cannot show them to you even by accident.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import pandas as pd

import taxonomy

# One letter per category, all distinct. Chosen to be guessable: conTract takes `t` because
# `c` belongs to clinical.
CATEGORY_KEYS = {"m": "merger", "d": "dilution", "c": "clinical", "e": "earnings",
                 "t": "contract", "g": "guidance", "l": "legal", "i": "insider",
                 "a": "analyst", "o": "other"}
MATERIALITY_KEYS = {"1": "low", "2": "medium", "3": "high"}
DIRECTION_KEYS = {"b": "bull", "r": "bear", "n": "neutral"}
KEYMAPS = {"category": CATEGORY_KEYS, "materiality": MATERIALITY_KEYS, "direction": DIRECTION_KEYS}

UNSURE = "unsure"
CONTROL = {"u": "mark unsure", "s": "skip for now", "p": "go back one row",
           "?": "show the definitions", "q": "save and quit"}


def getkey() -> str:
    """One keypress, no Enter. Falls back to a typed line where that is not possible."""
    try:
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):      # a function or arrow key sends two values
            msvcrt.getwch()
            return ""
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.lower()
    except ImportError:
        line = input()
        return line.strip()[:1].lower()


def definitions(field: str) -> str:
    """Pull this field's section out of the teacher's own prompt, so the two cannot drift."""
    heading = {"category": "CATEGORIES", "materiality": "MATERIALITY", "direction": "DIRECTION"}[field]
    text = taxonomy.SYSTEM_PROMPT
    start = text.index(heading)
    nxt = [text.index(h) for h in ("CATEGORIES", "MATERIALITY", "DIRECTION", "Rules:")
           if h in text and text.index(h) > start]
    return text[start:min(nxt)] if nxt else text[start:]


def legend(field: str) -> str:
    keys = KEYMAPS[field]
    pairs = [f"[{k}] {v}" for k, v in keys.items()]
    return "   ".join(pairs)


def save(df: pd.DataFrame, path: Path) -> None:
    """Write via a temporary file: an interrupted write cannot truncate the real sheet."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=Path("data/gold/gold_blank_pilot.csv"))
    ap.add_argument("--fields", default="category,materiality")
    ap.add_argument("--redo", action="store_true", help="revisit rows that already have answers")
    args = ap.parse_args()

    if not sys.stdin.isatty():
        raise SystemExit(
            "This needs an interactive terminal - it reads single keypresses.\n"
            "  Open a Terminal tab and run it there rather than through a Run button."
        )

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    for field in fields:
        if field not in KEYMAPS:
            raise SystemExit(f"no key bindings for `{field}`")

    df = pd.read_csv(args.gold, dtype=str)
    for field in fields:
        if field not in df.columns:
            df[field] = ""
        df[field] = df[field].fillna("")

    todo = [i for i in df.index if args.redo or any(df.at[i, f] == "" for f in fields)]
    if not todo:
        print(f"every row in {args.gold} already has {', '.join(fields)}. Nothing to do.")
        print("  Pass --redo to go through them again.")
        return

    print(f"\n{args.gold}  -  {len(todo)} of {len(df)} rows to label\n")
    print("  Judge the headline alone. The ticker and date are deliberately not shown: the")
    print("  teacher did not see them either.")
    print("  " + "   ".join(f"[{k}] {v}" for k, v in CONTROL.items()))
    print("\n  Answers save as you go. Quit any time and re-run to pick up where you left off.\n")

    position = 0
    while 0 <= position < len(todo):
        i = todo[position]
        headline = str(df.at[i, "Article_title"])
        print("=" * 78)
        print(f"  {position + 1} of {len(todo)}        {len(todo) - position - 1} after this")
        print()
        for line in textwrap.wrap(headline, width=74) or [""]:
            print(f"  {line}")
        print()

        stepped_back = False
        for field in fields:
            if df.at[i, field] and not args.redo:
                continue
            while True:
                print(f"  {field:11s} {legend(field)}")
                key = getkey()
                if key == "q":
                    save(df, args.gold)
                    done = sum(1 for n in df.index if all(df.at[n, f] for f in fields))
                    print(f"\n  saved. {done} of {len(df)} rows complete.\n")
                    return
                if key == "?":
                    print()
                    print("  " + definitions(field).replace("\n", "\n  "))
                    continue
                if key == "p":
                    position = max(0, position - 1)
                    # Clear the row being returned to, so it is answered afresh rather than
                    # skipped as already done.
                    for f in fields:
                        df.at[todo[position], f] = ""
                    stepped_back = True
                    break
                if key == "s":
                    break
                value = KEYMAPS[field].get(key) or (UNSURE if key == "u" else None)
                if value is None:
                    print(f"    '{key}' is not one of those keys")
                    continue
                df.at[i, field] = value
                print(f"    -> {value}")
                break
            if stepped_back:
                break

        if stepped_back:
            continue
        save(df, args.gold)
        position += 1

    save(df, args.gold)
    done = sum(1 for n in df.index if all(df.at[n, f] for f in fields))
    unsure = {f: int((df[f] == UNSURE).sum()) for f in fields}
    print("=" * 78)
    print(f"\n  finished. {done} of {len(df)} rows complete.")
    for field, n in unsure.items():
        if n:
            print(f"    {field}: {n} marked unsure (these are excluded from scoring)")
    print(f"\n  score it:\n    python code/score_gold.py --gold {args.gold} --tag full "
          f"--fields {','.join(fields)} --annotator \"hand-labelled\"\n")


if __name__ == "__main__":
    main()
