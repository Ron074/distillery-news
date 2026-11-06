"""M5: do the STUDENT's materiality tiers predict real abnormal returns, or only the teacher's?

    python code/m5_downstream.py --sample data/samples/fnspid_clean_48040_seed42.csv \
        --preds results/preds_3head.csv --wrds-dir <your-wrds-pull> --market-adjust

Every fidelity number in M3 measures agreement with the teacher. A student that copied a confident
mistake would score well on all of them. `validate_teacher.py` broke that loop for the teacher by
testing its labels against recorded prices; this does the same for the student, on the held-out
test split only - the rows it never trained on.

The question is not whether the student agrees with the teacher about materiality. It is whether a
headline the STUDENT calls `high` actually moves more than one it calls `low`, after taking the
market out and controlling for company size. If yes, the distillation carried real signal rather
than a mimicry of one.

Three things this does that the teacher check did not:

  * It runs both label sets over the SAME rows, so teacher and student are compared on identical
    headlines, identical prices, identical joins. Any difference is the labels.
  * It clusters standard errors by ticker. `validate_teacher.py` says in its own output that the
    Mann-Whitney p-value is optimistic because repeated tickers are not clustered - a single
    heavily-covered company can supply hundreds of correlated rows and inflate significance. The
    regression below fixes that with a CR1 sandwich estimator rather than noting the problem.
  * It runs a null arm. Labels shuffled within the joined data must collapse the effect to nothing.
    If a shuffled label still "predicts" returns, the pipeline is finding structure in the join and
    not in the labels, and every number here would be an artifact.

Reference prices are academically licensed, read from --wrds-dir, never copied into this
repository, and used for validation only - never as a training input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_teacher import CAP_BANDS, DAILY_FILE, default_wrds_dir, market_returns

TIERS = ("low", "medium", "high")


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS with cluster-robust (CR1) standard errors, clustered on `groups`.

    Written out rather than imported because statsmodels is not installed here, and because for a
    report the estimator is better visible than hidden. The meat of the sandwich sums each
    cluster's score contribution, so rows sharing a ticker are allowed to be correlated with each
    other in any way; the CR1 factor is the standard small-sample correction.
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    meat = np.zeros((k, k))
    uniq = np.unique(groups)
    for g in uniq:
        m = groups == g
        Xg, ug = X[m], resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)

    G = len(uniq)
    c = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = XtX_inv @ meat @ XtX_inv * c
    return beta, np.sqrt(np.maximum(np.diag(V), 0)), G


def design(df: pd.DataFrame, col: str) -> tuple[np.ndarray, list[str]]:
    """Intercept, two tier dummies (`low` is the reference), and cap-band dummies.

    Size is a confounder, not a nuisance: small companies move more whatever the news says, and
    the teacher tags small-company news `high` more often. Without the band dummies the tier
    coefficient would partly be measuring market cap.
    """
    cols = [np.ones(len(df))]
    names = ["intercept"]
    for tier in ("medium", "high"):
        cols.append((df[col] == tier).to_numpy(float))
        names.append(tier)
    for _, _, name in CAP_BANDS[1:]:               # first band is the reference
        cols.append((df["band"] == name).to_numpy(float))
        names.append(f"band:{name}")
    return np.column_stack(cols), names


def fit(df: pd.DataFrame, col: str, label: str, seed: int | None = None) -> dict:
    d = df
    if seed is not None:
        d = df.copy()
        d[col] = np.random.default_rng(seed).permutation(d[col].to_numpy())
    X, names = design(d, col)
    beta, se, G = ols_cluster(d["abs_ret"].to_numpy(float), X, d["ticker"].to_numpy())
    out = {"label_source": label, "n": int(len(d)), "clusters": G, "terms": {}}
    from scipy import stats
    for name, b, s in zip(names, beta, se):
        t = b / s if s > 0 else float("nan")
        p = float(2 * stats.t.sf(abs(t), df=max(G - 1, 1))) if s > 0 else float("nan")
        out["terms"][name] = {"coef": round(float(b), 4), "se": round(float(s), 4),
                              "t": round(float(t), 3), "p": p}
    med = d.groupby(col)["abs_ret"].median()
    out["median_abs_ret"] = {t: round(float(med.get(t, np.nan)), 3) for t in TIERS}
    out["n_by_tier"] = {t: int((d[col] == t).sum()) for t in TIERS}
    return out


def band_of(cap: float) -> str:
    for lo, hi, name in CAP_BANDS:
        if lo <= cap < hi:
            return name
    return "unbanded"


def build(args) -> pd.DataFrame:
    preds = pd.read_csv(args.preds, dtype=str)
    need = {"uid", "student_materiality", "teacher_materiality"}
    if not need <= set(preds.columns):
        raise SystemExit(f"{args.preds} needs {sorted(need)}; found {list(preds.columns)}")
    print(f"predictions: {len(preds):,} held-out rows from {args.preds.name}")

    meta = pd.read_csv(args.sample, dtype=str)[["uid", "Date", "Stock_symbol"]]
    df = preds.merge(meta, on="uid", how="inner")
    df["news_date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True
                                     ).dt.tz_localize(None).dt.normalize()
    df["ticker"] = df["Stock_symbol"].str.upper().str.strip()
    df = df.dropna(subset=["news_date", "ticker"])
    print(f"  joined to sample metadata: {len(df):,}")

    path = Path(args.wrds_dir or default_wrds_dir() or ".") / DAILY_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found; pass --wrds-dir")
    daily = pd.read_parquet(path, columns=["Ticker", "DlyCalDt", "DlyRet", "DlyCap"])
    daily["ticker"] = daily["Ticker"].astype(str).str.upper().str.strip()
    daily["news_date"] = pd.to_datetime(daily["DlyCalDt"], errors="coerce").dt.normalize()
    daily = daily[daily["ticker"].isin(set(df["ticker"]))]

    j = df.merge(daily[["ticker", "news_date", "DlyRet", "DlyCap"]], on=["ticker", "news_date"])
    # A ticker can belong to more than one company on a date - dual share classes, or a symbol
    # reassigned after a delisting. Such a headline matches several firms' returns with no way to
    # tell which the news was about, so drop it rather than double-count it.
    amb = j["uid"].duplicated(keep=False)
    if amb.any():
        print(f"  dropped {int(j.loc[amb, 'uid'].nunique()):,} headlines whose ticker matched more "
              "than one company that day")
        j = j[~amb]
    j["ret"] = pd.to_numeric(j["DlyRet"], errors="coerce")
    j["cap"] = pd.to_numeric(j["DlyCap"], errors="coerce")
    j = j.dropna(subset=["ret", "cap"])

    if args.market_adjust:
        mkt = market_returns(path, j["news_date"].min().date(), j["news_date"].max().date())
        j = j.join(mkt, on="news_date")
        n_missing = int(j["mkt_ret"].isna().sum())
        if n_missing:
            print(f"  note: {n_missing:,} rows had no market return and are dropped")
        j = j.dropna(subset=["mkt_ret"])
        j["ret"] = j["ret"] - j["mkt_ret"]
    j["abs_ret"] = j["ret"].abs() * 100
    j["band"] = j["cap"].map(band_of)
    print(f"  joined to a trading day: {len(j):,}  "
          f"({j['ticker'].nunique():,} tickers, {j['news_date'].nunique():,} days)")
    return j


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, default=Path("results/preds_3head.csv"))
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--wrds-dir", type=Path, default=None)
    ap.add_argument("--market-adjust", action="store_true", default=True)
    ap.add_argument("--raw-returns", dest="market_adjust", action="store_false",
                    help="do not subtract the market; credits the news for market-wide days")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/m5_downstream.json"))
    args = ap.parse_args()

    j = build(args)
    measure = "|abnormal return| %" if args.market_adjust else "|return| %"

    runs = {
        "teacher": fit(j, "teacher_materiality", "teacher labels (claude-opus-5)"),
        "student": fit(j, "student_materiality", "student predictions (FinBERT, held-out)"),
        "student_shuffled": fit(j, "student_materiality", "student labels shuffled (null arm)",
                                seed=args.seed),
    }

    print(f"\n{measure} on the news day, by materiality tier (median):")
    print(f"  {'source':24s} {'low':>8s} {'medium':>8s} {'high':>8s}")
    for key in ("teacher", "student"):
        m = runs[key]["median_abs_ret"]
        print(f"  {key:24s} {m['low']:>8.2f} {m['medium']:>8.2f} {m['high']:>8.2f}")

    print(f"\nOLS: {measure} ~ tier + market-cap band, SEs clustered by ticker")
    print(f"  {'source':20s} {'high vs low':>14s} {'se':>8s} {'t':>8s} {'p':>11s}")
    for key, run in runs.items():
        h = run["terms"]["high"]
        print(f"  {key:20s} {h['coef']:>+13.3f}pp {h['se']:>8.3f} {h['t']:>8.2f} {h['p']:>11.2e}")
    print(f"\n  medium vs low, for comparison:")
    for key in ("teacher", "student"):
        m = runs[key]["terms"]["medium"]
        print(f"    {key:18s} {m['coef']:>+7.3f}pp  t={m['t']:>6.2f}  p={m['p']:.2e}")

    t_h = runs["teacher"]["terms"]["high"]["coef"]
    s_h = runs["student"]["terms"]["high"]["coef"]
    recovered = s_h / t_h if t_h else float("nan")
    print(f"\n  clusters (distinct tickers): {runs['student']['clusters']:,}")
    print(f"  the student recovers {recovered:.0%} of the teacher's high-vs-low effect")

    null_t = runs["student_shuffled"]["terms"]["high"]["t"]
    verdict = "collapses as it must" if abs(null_t) < 3 else "DOES NOT COLLAPSE - investigate"
    print(f"  null arm (labels shuffled): t = {null_t:+.2f}  -> {verdict}")

    print("\nCAVEAT: a same-day association, not causation. The tier is measured against the move")
    print("on the day the headline is dated, so it cannot separate news that moved the stock from")
    print("coverage that followed a move. Reactive 'why is X up' pieces are tagged low and move a")
    print("lot, which pushes against this result rather than for it.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "check": "m5_downstream", "measure": measure, "n_joined": int(len(j)),
        "n_tickers": int(j["ticker"].nunique()), "n_days": int(j["news_date"].nunique()),
        "population": "held-out test split only",
        "se": "CR1 cluster-robust, clustered by ticker",
        "controls": "market-cap band dummies, first band as reference",
        "student_recovers_fraction_of_teacher_effect": round(float(recovered), 4),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
