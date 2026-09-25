"""M3 - distil the teacher into a FinBERT student and measure fidelity.

One shared encoder with three classification heads (category, materiality, direction), rather than
three separate models: the tasks are correlated, it trains once instead of three times, and it is
what would actually be deployed.

    python code/train_student.py --labels full --sample data/samples/<clean>.csv
    python code/train_student.py --labels full --sample ... --no-direction      # the ablation
    python code/train_student.py --labels full --sample ... --limit 5000        # learning curve

Fidelity here means agreement with the teacher, not correctness. The teacher agrees with itself
93% of the time, so that is the ceiling - report it beside any fidelity number.

Designed to run on a free Colab GPU. On CPU it works but takes hours per run.
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
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score

import taxonomy
from m1_reproduce import default_results_dir, set_seed
from teacher_label import DUP_SUFFIX, default_data_dir, labels_path

HEADS = {
    "category": taxonomy.CATEGORIES,
    "materiality": taxonomy.MATERIALITY,
    "direction": taxonomy.DIRECTION,
}


def load_dataset(labels_tag: str, sample: Path, data_dir: Path) -> pd.DataFrame:
    path = labels_path(data_dir, labels_tag)
    if not path.exists():
        raise SystemExit(f"no labels at {path}; run teacher_label.py first")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[~df["uid"].str.endswith(DUP_SUFFIX)]

    meta = pd.read_csv(sample, dtype=str)[["uid", "Date", "Article_title", "Stock_symbol"]]
    df = df.merge(meta, on="uid", how="inner")
    df["year"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.year
    df = df.dropna(subset=["year", "Article_title"])
    df["year"] = df["year"].astype(int)
    return df


def drop_near_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Templated headlines repeat across the corpus. Left in, near-identical items land on both
    sides of the split and fidelity is inflated by recognition rather than learning."""
    key = (df["Article_title"].str.lower()
           .str.replace(r"[0-9]+", "#", regex=True)
           .str.replace(r"[^a-z# ]", "", regex=True)
           .str.replace(r"\s+", " ", regex=True).str.strip())
    before = len(df)
    df = df.loc[~key.duplicated()].copy()
    return df, before - len(df)


def split_by_time(df: pd.DataFrame, val_year: int, test_from: int):
    """Fully temporal: the model is asked to generalise forward, never to interpolate."""
    train = df[df["year"] < val_year]
    val = df[df["year"] == val_year]
    test = df[df["year"] >= test_from]
    return train, val, test


def class_weights(labels: list[str], names: list[str]):
    counts = pd.Series(labels).value_counts().reindex(names).fillna(0).to_numpy()
    # Inverse frequency, normalised to mean 1. `high` materiality is ~3% of the corpus; unweighted,
    # a model that never predicts it still scores well.
    with np.errstate(divide="ignore"):
        w = np.where(counts > 0, len(labels) / (len(names) * np.maximum(counts, 1)), 0.0)
    w = w / w[w > 0].mean()
    return torch.tensor(w, dtype=torch.float)


def load_encoder(name: str):
    """AutoModel, with the same fallback M1 needed: yiyanghkust/finbert-pretrain omits the
    `model_type` key, so the Auto registry cannot dispatch. Only name the BERT class once the
    config confirms a Bert architecture, so an unsupported checkpoint still fails loudly instead
    of loading weights into a mismatched class."""
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(name)
    except ValueError:
        from transformers import BertConfig, BertModel

        config = BertConfig.from_pretrained(name)
        if not any(arch.startswith("Bert") for arch in (config.architectures or [])):
            raise
        return BertModel.from_pretrained(name)


class Student(nn.Module):
    def __init__(self, encoder_name: str, heads: dict[str, list[str]]):
        super().__init__()
        self.encoder = load_encoder(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, len(values))
                                    for name, values in heads.items()})

    def forward(self, **enc):
        out = self.encoder(**enc)
        # Mean-pool over real tokens: robust across checkpoints that lack a trained pooler.
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        return {name: head(pooled) for name, head in self.heads.items()}


def encode_batch(tokenizer, texts, max_length, device):
    enc = tokenizer(list(texts), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict(model, tokenizer, texts, heads, device, batch_size, max_length):
    model.eval()
    out = {name: [] for name in heads}
    for i in range(0, len(texts), batch_size):
        logits = model(**encode_batch(tokenizer, texts[i:i + batch_size], max_length, device))
        for name in heads:
            out[name].extend(logits[name].argmax(-1).cpu().tolist())
    return out


def evaluate(model, tokenizer, frame, heads, device, batch_size, max_length) -> dict:
    preds = predict(model, tokenizer, frame["Article_title"].tolist(), heads, device,
                    batch_size, max_length)
    scores = {}
    for name, names in heads.items():
        truth = [names.index(v) for v in frame[name]]
        pred = preds[name]
        scores[name] = {
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "per_class": classification_report(truth, pred, labels=list(range(len(names))),
                                               target_names=names, output_dict=True,
                                               zero_division=0),
        }
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="full", help="label run tag to train on")
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--encoder", default="yiyanghkust/finbert-pretrain")
    ap.add_argument("--no-direction", action="store_true",
                    help="drop the direction head, to measure whether it helps or hurts the others")
    ap.add_argument("--val-year", type=int, default=2018)
    ap.add_argument("--test-from", type=int, default=2019)
    ap.add_argument("--limit", type=int, default=None, help="cap training rows, for the learning curve")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--save-predictions", type=Path, default=None,
                    help="write per-headline student and teacher labels for the test split (M5 needs these)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from m1_reproduce import load_tokenizer

    set_seed(args.seed)
    heads = {k: v for k, v in HEADS.items() if not (args.no_direction and k == "direction")}
    data_dir = args.data_dir or default_data_dir()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = load_dataset(args.labels, args.sample, data_dir)
    df, dropped = drop_near_duplicates(df)
    train, val, test = split_by_time(df, args.val_year, args.test_from)
    if args.limit:
        train = train.sample(n=min(args.limit, len(train)), random_state=args.seed)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        if frame.empty:
            raise SystemExit(f"{name} split is empty; check --val-year / --test-from")

    print(f"dropped {dropped:,} near-duplicate headlines")
    print(f"train {len(train):,} (<{args.val_year})  val {len(val):,} ({args.val_year})  "
          f"test {len(test):,} ({args.test_from}+)")
    print(f"heads: {', '.join(heads)}  |  encoder {args.encoder}  |  device {device}", flush=True)

    tokenizer = load_tokenizer(args.encoder)
    model = Student(args.encoder, heads).to(device)
    weights = {n: class_weights(train[n].tolist(), names).to(device) for n, names in heads.items()}
    losses = {n: torch.nn.CrossEntropyLoss(weight=weights[n]) for n in heads}

    rows = list(zip(train["Article_title"], *[train[n] for n in heads]))

    def collate(batch):
        texts = [b[0] for b in batch]
        enc = encode_batch(tokenizer, texts, args.max_length, device)
        for i, name in enumerate(heads, start=1):
            enc[f"y_{name}"] = torch.tensor([heads[name].index(b[i]) for b in batch], device=device)
        return enc

    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)
    total = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)

    best_val, best_state, best_epoch = -1.0, None, -1
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            targets = {n: batch.pop(f"y_{n}") for n in heads}
            logits = model(**batch)
            loss = sum(losses[n](logits[n], targets[n]) for n in heads)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

        scores = evaluate(model, tokenizer, val, heads, device, args.batch_size, args.max_length)
        mean_f1 = float(np.mean([scores[n]["macro_f1"] for n in heads]))
        print(f"epoch {epoch}: train loss {running / len(loader):.4f}  val mean macro-F1 {mean_f1:.4f}  "
              + "  ".join(f"{n}={scores[n]['macro_f1']:.3f}" for n in heads), flush=True)
        if mean_f1 > best_val:
            best_val, best_epoch = mean_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - t0

    model.load_state_dict(best_state)
    model.to(device)
    t0 = time.perf_counter()
    test_scores = evaluate(model, tokenizer, test, heads, device, args.batch_size, args.max_length)
    infer_seconds = time.perf_counter() - t0

    if args.save_predictions:
        # M5 joins the student's per-headline calls to realised price moves, which aggregate
        # metrics cannot support. Saved keyed by uid so it joins back to the corpus.
        preds = predict(model, tokenizer, test["Article_title"].tolist(), heads, device,
                        args.batch_size, args.max_length)
        out = test[["uid"]].copy()
        for name, names in heads.items():
            out[f"student_{name}"] = [names[i] for i in preds[name]]
            out[f"teacher_{name}"] = test[name].values
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.save_predictions, index=False)
        print(f"saved {len(out):,} test predictions -> {args.save_predictions}")

    run = {
        "milestone": "M3", "encoder": args.encoder, "heads": list(heads),
        "labels_tag": args.labels, "seed": args.seed, "epochs": args.epochs,
        "best_epoch": best_epoch, "best_val_mean_macro_f1": best_val,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "near_duplicates_dropped": int(dropped),
        "split": {"train_before": args.val_year, "val_year": args.val_year, "test_from": args.test_from},
        "train_seconds": train_seconds, "infer_seconds": infer_seconds,
        "ms_per_headline": infer_seconds / max(len(test), 1) * 1000,
        "device": device, "test": test_scores,
        "teacher_self_consistency_note": "teacher agrees with itself 93.0% on all three fields; "
                                         "that is the ceiling on fidelity",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.machine()} python {platform.python_version()}",
    }

    print("\n" + "=" * 68)
    print(f"M3 fidelity vs teacher - test {args.test_from}+ (n={len(test):,})")
    print("=" * 68)
    for name in heads:
        s = test_scores[name]
        print(f"  {name:12s} accuracy {s['accuracy']:.4f}   macro-F1 {s['macro_f1']:.4f}")
    if "materiality" in test_scores:
        high = test_scores["materiality"]["per_class"].get("high", {})
        print(f"\n  high-materiality recall {high.get('recall', float('nan')):.4f} "
              f"on {int(high.get('support', 0))} examples  <- the class that matters most, and the rarest")
    print(f"\n  inference {run['ms_per_headline']:.2f} ms/headline on {device}")

    results_dir = args.results_dir or default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("no_direction" if args.no_direction else "3head")
    if args.limit:
        tag += f"_n{args.limit}"
    (results_dir / f"m3_{tag}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nwrote {results_dir / f'm3_{tag}.json'}")

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_dir / "student.pt")
        tokenizer.save_pretrained(args.save_dir)
        (args.save_dir / "heads.json").write_text(json.dumps(heads), encoding="utf-8")
        print(f"saved student -> {args.save_dir}")


if __name__ == "__main__":
    main()
