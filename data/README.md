# data/

How to obtain the datasets. **Raw data is git-ignored** (large and/or licensed) — this folder documents how to
regenerate it, not the data itself.

## Public (used for training/distillation)
- **Financial PhraseBank** — financial-sentiment labels, for reproducing the FinBERT baseline (M1).
- **FNSPID** — public financial-news headlines + tickers + timestamps; the distillation corpus (M2–M3).

## Validation-only (not training data)
- **Academic price data (e.g., CRSP via WRDS)** — used *only* at the downstream step (M5) to turn each headline
  into a realized move + a size control. **Academic-licensed → never committed here.** Access is via your own
  institutional subscription; place any local pulls under `data/wrds/` (git-ignored) and reference by path.

No raw dataset files are stored in this repository.
