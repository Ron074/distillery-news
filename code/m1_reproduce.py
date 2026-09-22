"""M1 — reproduce the FinBERT sentiment baseline on Financial PhraseBank.

Three arms, all evaluated on the same held-out test split:

  baseline  TF-IDF + logistic regression      (does the transformer earn its complexity?)
  zeroshot  an already-fine-tuned checkpoint  (ProsusAI/finbert, off the shelf)
  finetune  fine-tune a base checkpoint       (the actual reproduction)

Examples:
    python code/m1_reproduce.py --arm baseline
    python code/m1_reproduce.py --arm zeroshot --model ProsusAI/finbert
    python code/m1_reproduce.py --arm finetune --model bert-base-uncased --epochs 4
    python code/m1_reproduce.py --arm finetune --model yiyanghkust/finbert-pretrain --epochs 4

Paths come from --data-dir / --results-dir or $DISTILLERY_DATA_DIR / $DISTILLERY_RESULTS_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import phrasebank
from phrasebank import LABELS


def default_results_dir() -> Path:
    env = os.environ.get("DISTILLERY_RESULTS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": classification_report(
            y_true, y_pred, target_names=LABELS, labels=list(range(len(LABELS))),
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }


def report_text(run: dict) -> str:
    m = run["metrics"]
    lines = [
        "=" * 72,
        f"M1 - Financial PhraseBank ({run['agreement']})  |  arm: {run['arm']}",
        f"model: {run['model']}",
        f"run:   {run['timestamp']}  seed={run['seed']}  device={run['device']}",
        "=" * 72,
        "",
        f"test n = {run['n_test']}   (train {run['n_train']} / val {run['n_val']})",
        "",
        f"  accuracy     {m['accuracy']:.4f}",
        f"  macro F1     {m['macro_f1']:.4f}",
        f"  weighted F1  {m['weighted_f1']:.4f}",
        "",
        "per class:",
        f"  {'label':10s} {'prec':>7s} {'rec':>7s} {'f1':>7s} {'n':>6s}",
    ]
    for name in LABELS:
        c = m["per_class"][name]
        lines.append(f"  {name:10s} {c['precision']:7.4f} {c['recall']:7.4f} {c['f1-score']:7.4f} {int(c['support']):6d}")
    lines += ["", "confusion matrix (rows = true, cols = pred; order: " + ", ".join(LABELS) + "):"]
    for name, row in zip(LABELS, m["confusion_matrix"]):
        lines.append(f"  {name:10s} " + " ".join(f"{v:5d}" for v in row))
    if run.get("train_seconds") is not None:
        epochs = f" over {run['epochs']} epochs" if run.get("epochs") else ""
        lines += ["", f"train wall-clock: {run['train_seconds']:.1f}s{epochs}"]
    if run.get("infer_seconds") is not None:
        n = run["n_test"]
        lines.append(f"inference: {run['infer_seconds']:.2f}s for {n} sentences "
                     f"({run['infer_seconds'] / n * 1000:.2f} ms/sentence)")
    if run.get("notes"):
        lines += ["", "NOTE: " + run["notes"]]
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- arms

def run_baseline(train, val, test, args) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
    )
    t0 = time.perf_counter()
    pipe.fit(train["sentence"], train["label_id"])
    train_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = pipe.predict(test["sentence"])
    infer_seconds = time.perf_counter() - t0

    return {
        "model": "tfidf(1,2) + logistic regression",
        "metrics": score(test["label_id"], pred),
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "epochs": None,
        "device": "cpu",
    }


def _predict_transformer(model, tokenizer, sentences, device, batch_size, max_length):
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = list(sentences[i : i + batch_size])
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def _checkpoint_label_permutation(model) -> list[int] | None:
    """Map a checkpoint's own label order onto phrasebank.LABELS. None if it has no real labels."""
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if not all(name in LABELS for name in id2label.values()):
        return None
    # position i of the returned list = the checkpoint's index for LABELS[i]
    label2ckpt = {v: k for k, v in id2label.items()}
    return [label2ckpt[name] for name in LABELS]


def run_zeroshot(train, val, test, args) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    perm = _checkpoint_label_permutation(model)
    if perm is None:
        raise SystemExit(
            f"{args.model} has no negative/neutral/positive head (id2label={model.config.id2label}); "
            "use --arm finetune for a base checkpoint."
        )

    t0 = time.perf_counter()
    raw_pred = _predict_transformer(model, tokenizer, test["sentence"].tolist(), device, args.batch_size, args.max_length)
    infer_seconds = time.perf_counter() - t0

    ckpt2ours = {ckpt_idx: ours for ours, ckpt_idx in enumerate(perm)}
    pred = [ckpt2ours[p] for p in raw_pred]

    return {
        "model": args.model,
        "metrics": score(test["label_id"], pred),
        "train_seconds": None,
        "infer_seconds": infer_seconds,
        "epochs": None,
        "device": device,
        "checkpoint_id2label": {int(k): str(v) for k, v in model.config.id2label.items()},
        "notes": (
            "Off-the-shelf checkpoint, NOT a clean held-out score: ProsusAI/finbert was itself fine-tuned on "
            "Financial PhraseBank, so these test sentences were very likely in its training data. Treat as a "
            "pipeline/label-mapping sanity check and an upper reference, not as a reproduction."
        ),
    }


def run_finetune(train, val, test, args) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id=dict(phrasebank.LABEL2ID),
    ).to(device)

    def collate(rows):
        texts = [r[0] for r in rows]
        labels = torch.tensor([r[1] for r in rows])
        enc = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        enc["labels"] = labels
        return enc

    train_rows = list(zip(train["sentence"], train["label_id"]))
    loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_val_f1, best_state, best_epoch = -1.0, None, -1
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

        val_pred = _predict_transformer(model, tokenizer, val["sentence"].tolist(), device, args.batch_size, args.max_length)
        val_f1 = f1_score(val["label_id"], val_pred, average="macro")
        print(f"epoch {epoch}: train loss {running / len(loader):.4f}  val macro-F1 {val_f1:.4f}", flush=True)
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = float(val_f1), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - t0

    model.load_state_dict(best_state)
    model.to(device)
    t0 = time.perf_counter()
    pred = _predict_transformer(model, tokenizer, test["sentence"].tolist(), device, args.batch_size, args.max_length)
    infer_seconds = time.perf_counter() - t0

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"saved best checkpoint -> {save_dir}")

    return {
        "model": args.model,
        "metrics": score(test["label_id"], pred),
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "epochs": args.epochs,
        "device": device,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "hyperparams": {"lr": args.lr, "batch_size": args.batch_size, "max_length": args.max_length,
                        "weight_decay": args.weight_decay, "warmup_frac": 0.1},
    }


ARMS = {"baseline": run_baseline, "zeroshot": run_zeroshot, "finetune": run_finetune}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", default=None, help="HF checkpoint (ignored by --arm baseline)")
    ap.add_argument("--agreement", default="50agree", choices=sorted(phrasebank.AGREEMENTS))
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--save-dir", type=Path, default=None, help="where to write the fine-tuned checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    ap.add_argument("--tag", default=None, help="suffix for the results filename")
    args = ap.parse_args()

    if args.arm != "baseline" and not args.model:
        args.model = "ProsusAI/finbert" if args.arm == "zeroshot" else "bert-base-uncased"

    set_seed(args.seed)
    df = phrasebank.load(args.agreement, args.data_dir)
    train, val, test = phrasebank.split(df, seed=args.seed)
    print(f"{args.arm}: train={len(train)} val={len(val)} test={len(test)}", flush=True)

    run = ARMS[args.arm](train, val, test, args)
    run.update({
        "arm": args.arm,
        "agreement": args.agreement,
        "seed": args.seed,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.machine()} python {platform.python_version()}",
    })

    text = report_text(run)
    print("\n" + text)

    results_dir = args.results_dir or default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (args.model.split("/")[-1] if args.model else "tfidf")
    stem = f"m1_{args.arm}_{tag}".replace(" ", "-")
    (results_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    (results_dir / f"{stem}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"wrote {results_dir / (stem + '.txt')}")


if __name__ == "__main__":
    main()
