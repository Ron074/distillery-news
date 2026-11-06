# results/

Generated tables and figures, populated **as milestones complete** — nothing is reported here until the
corresponding step is done (no placeholder or fabricated numbers). Expected artifacts:

- `fidelity.txt` — student-vs-teacher agreement (accuracy / macro-F1, confusion) — M3
- `cost_latency.txt` — $ and ms per 1k headlines, LLM vs student vs baselines — M4
- `downstream.txt` — do the student's labels predict real moves as well as the teacher's — M5
- `ablations.txt` — learning curve, FinBERT vs DistilBERT, headline vs full-article — M3/M6

## Provenance of the price-validated files

`m2_earnings_check.json`, `m2_materiality_check.json` and `m5_downstream.json` are the only files
here computed from **academically licensed reference data** (CRSP daily returns and Compustat
announcement dates, obtained through the author's institutional WRDS subscription).

What that means for this repository:

- **No licensed data is redistributed here.** These three files hold aggregate statistics only —
  counts, medians, regression coefficients and p-values pooled over thousands of headlines and
  hundreds of companies. There is no identifier of any kind in them: no PERMNO, GVKEY, CUSIP,
  ticker symbol, date, or per-observation return, and no way to recover one from what is reported.
- **The licensed data is used for validation only, never as a training input.** The student learns
  from public news headlines and the teacher's labels. Prices are joined afterwards, to ask whether
  the labels correspond to anything real; nothing in the model was fitted to them.
- **The pull itself is never committed.** It is read at runtime from `--wrds-dir` or
  `$DISTILLERY_WRDS_DIR`, and `data/wrds/`, `*.parquet` and `*.csv` are all git-ignored. No path to
  it appears in any source file.

Reporting aggregate results of research is what the academic licence is for; redistributing the
underlying data is not, and is not done. Anyone wishing to reproduce these three files needs their
own WRDS access — see `data/README.md`.
