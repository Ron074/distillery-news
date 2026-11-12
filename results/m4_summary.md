# M4 - is the student worth its cost? A TF-IDF baseline on the distillation task

M1's TF-IDF number is PhraseBank sentiment and says nothing about this task. This is the same kind of baseline - TF-IDF unigrams and bigrams into one logistic regression per head - trained on **exactly the student's rows**: 26,976 train, 5,953 validation (2018), 8,591 test (2019 onward), after the same 1,776 near-duplicates were dropped. The script checks this rather than assuming it: row counts against `m3_3head.json`, the test uid set against `preds_3head.csv`, and the teacher label on every test row against the same file. All three matched.

Set up not to lose on a technicality: balanced class weights (the student's loss is inverse-frequency weighted too), and the regularisation strength C chosen per head on the 2018 validation year from 0.25, 1.0, 4.0, 16.0, 64.0, 256.0 - the chosen values (category 16, materiality 4, direction 4) are all inside the grid, not on its edge.

## Student against baseline, same 8,591 test rows

| head | majority | TF-IDF + LR | student (M3) | student - baseline, accuracy [95% CI] | share of reachable gap: baseline / student |
|---|---|---|---|---|---|
| `category` | 0.496 | 0.8455 | 0.8611 | +0.016 [+0.008, +0.024] | 75% / 78% |
| `materiality` | 0.731 | 0.8717 | 0.8716 | -0.000 [-0.008, +0.008] | 58% / 58% |
| `direction` | 0.664 | 0.8044 | 0.8497 | +0.045 [+0.037, +0.054] | 46% / 61% |

| head | baseline macro-F1 | student macro-F1 | student - baseline [95% CI] |
|---|---|---|---|
| `category` | 0.8090 | 0.8292 | +0.020 [+0.007, +0.034] |
| `materiality` | 0.7034 | 0.7173 | +0.014 [-0.007, +0.034] |
| `direction` | 0.7444 | 0.8174 | +0.073 [+0.062, +0.085] |

Intervals resample whole tickers (2,915 of them, 2,000 draws), not rows - headlines about one company are correlated, the same reason M5 clusters by ticker. The ceiling for 'share of reachable gap' is the teacher's self-consistency (category 96.2%, materiality 97.4%, direction 97.0%).

## `high` materiality, the class the project exists for

| | recall | precision | flagged |
|---|---|---|---|
| student (M3) | **0.791** | 0.329 | 566 |
| baseline, default threshold | 0.464 | 0.388 | 281 |
| baseline, best threshold at the student's precision | 0.579 | 0.333 | 408 |

Support: 235 `high` headlines in test. The two models sit at different operating points by default, so the third row gives the baseline its best possible threshold at the student's precision, **chosen on the test set itself** - an advantage the student did not get. The student still catches 21 percentage points more of the market-movers. A threshold-free comparison (average precision) needs the student's probabilities, which the M3 run did not keep.

## Null arm

| head | baseline | labels shuffled | majority |
|---|---|---|---|
| `category` | 0.8455 | 0.2610 | 0.4963 |
| `materiality` | 0.8717 | 0.5202 | 0.7306 |
| `direction` | 0.8044 | 0.4193 | 0.6644 |

Same pipeline and C, training labels permuted. Every head collapses to or below the majority rate (below, because balanced weighting spreads guesses across rare classes), so the baseline's scores come from the headlines, not from the split.

## What it means

- **The student's margin is narrower than the M3 table alone suggests.** On category accuracy a linear model on word counts gets within 1.6 points; on materiality accuracy the two are indistinguishable. Much of the teacher's judgement on these two fields is recoverable from vocabulary.
- **Where the student earns its cost is the hard part:** direction (+4.5 points accuracy, +0.073 macro-F1), and the rare `high` tier (79% recall against the baseline's best 58% at equal precision). A plausible reading, not yet tested: direction turns on word order and negation, and `high` on finding a rare class at all - both places a bag of words is weak. An error analysis of the rows where the two models disagree would check it.
- Materiality accuracy is the wrong lens for this project: it is 73% `low`, so both models score on the easy majority.

Baseline speed on this laptop's CPU, all three heads: 0.016 ms per headline batched, 0.89 ms one at a time (median of 5 repeats; single runs varied up to 3x). The full cost table - teacher, student, baseline - is the second half of M4.

Source: `m4_baseline.json` (2026-11-12T21:58:34+00:00, Windows AMD64 python 3.14.5). Regenerate with `python code/m4_summary.py`.
