# M5 — do the student's labels predict real moves, or only the teacher's?

Every number in M3 measures agreement with the teacher. A student that faithfully copied a
confident mistake would score well on all of them. M2 broke that loop for the *teacher* by testing
its labels against recorded prices; M5 does it for the *student*.

**Population:** the held-out test split only — 8,591 rows the student never trained on. 7,694
joined to a trading day, across **2,715 tickers** and 364 trading days. 38 headlines were dropped
because their ticker belonged to more than one company that day (dual share classes, or a symbol
reassigned after a delisting), since there is no way to tell which company the news was about.

**Specification:** `|abnormal return| % ~ materiality tier + market-cap band`, ordinary least
squares, with the first cap band as reference and `low` as the reference tier. Returns are
market-adjusted by subtracting the value-weighted return of every stock in the reference universe
that day, so a market-wide move is not credited to the headline. Cap bands are controls, not
nuisance: small companies move more whatever the news says, and the teacher tags small-company news
`high` more often, so without them the tier coefficient would partly be measuring size.

**Standard errors are clustered by ticker.** M2's Mann-Whitney p-value said in its own output that
it was optimistic because repeated tickers were not clustered — a single heavily-covered company
can supply hundreds of correlated rows. The CR1 sandwich estimator used here was checked against
simulation first: on cluster-assigned treatment with a large cluster error component it recovered
0.97× the estimator's true sampling standard deviation, where the naive OLS standard error gave
0.16×, i.e. **6× overconfident**.

## Result

| label source | high vs low | se | t | p |
|---|---|---|---|---|
| teacher (`claude-opus-5`) | **+5.56 pp** | 0.906 | 6.13 | 9.7e-10 |
| **student** (FinBERT, held-out) | **+2.63 pp** | 0.414 | **6.36** | 2.4e-10 |
| student, labels shuffled | +0.53 pp | 0.426 | 1.23 | 0.22 |

`medium` vs `low`: teacher **+1.11 pp** (t = 6.25), student **+0.83 pp** (t = 5.43).

**The student's own materiality labels predict real abnormal returns.** A headline the student
calls `high` moves 2.63 percentage points more, in absolute terms, than one it calls `low` — after
removing the market, controlling for company size, and allowing every row sharing a ticker to be
arbitrarily correlated. This is the one claim M3's fidelity figures structurally cannot make.

**The null arm collapses**, as it must: labels shuffled within the joined data give t = 1.23,
p = 0.22. Had a permuted label still "predicted" returns, the effect would have been an artifact of
the join rather than a property of the labels, and every figure above would be void.

## The student recovers 47% of the teacher's effect, and the reason is measurable

| | low | medium | high | median `high` move |
|---|---|---|---|---|
| teacher | 5,448 | 2,022 | **224** | 2.35% |
| student | 5,181 | 1,973 | **540** | 1.71% |

Not a weaker signal — a **diluted** one. The student calls `high` 2.4× as often as the teacher, and
the 316 extra headlines move less, which drags the tier's average move down from 2.35% to 1.71%.
Its `high` set is larger and softer rather than wrong.

That is a thresholding artifact, not a training failure, and it motivates M6 directly: routing by
confidence should sharpen the materiality tier as well as catch ambiguous direction calls. **M5
leaves M6 a number to beat — 47%.** Testing it requires per-row materiality probabilities, which
`preds_3head.csv` does not carry, so it needs one inference pass with probabilities retained.

## Caveat

A same-day association, not causation. The tier is measured against the move on the day the
headline is dated, so this cannot separate news that moved a stock from coverage that followed a
move. The direction of that bias is worth stating: reactive "why is X up today" pieces are tagged
`low` and move a great deal, which pushes **against** the result rather than for it.

Reference data is academically licensed and used for validation only — see the provenance note in
`results/README.md`.
