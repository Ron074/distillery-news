# M3 - distilled student vs teacher

One FinBERT encoder with three heads, trained on 26,976 teacher-labelled headlines and tested on **8,591 headlines from 2019 onward** - a period the model never saw. Near-duplicate headlines were removed before splitting (1,776 dropped), so no templated item sits on both sides.

**Fidelity means agreement with the teacher, not correctness.** The teacher agrees with itself 93.0% of the time across all three fields, so that is the ceiling; a student at 86% is closing most of the reachable gap rather than falling 14 points short of perfect.

## Main model

| head | majority baseline | student accuracy | macro-F1 | teacher self-consistency | share of reachable gap |
|---|---|---|---|---|---|
| `category` | 49.6% | **0.8611** | 0.8292 | 98.6% | 75% |
| `materiality` | 73.1% | **0.8716** | 0.7173 | 97.2% | 58% |
| `direction` | 66.4% | **0.8497** | 0.8174 | 96.6% | 61% |

**`high` materiality: recall 0.791, precision 0.329** on 235 test examples. It is 3.3% of the corpus and the single class the project exists to predict, so it is reported separately - macro-F1 averages it away, and without class weighting a model that never predicted it would still score well.

Recall and precision are far apart, and that is a deliberate consequence rather than a defect. Class weighting pushes the model to find the rare class, so it catches 79% of genuine high-materiality headlines while only 33% of the ones it flags turn out to be high - roughly two false alarms per real hit. For a screening tool that ranks what a person should read next, missing a market-moving item costs more than over-flagging a routine one, so this is the right side to err on. The unweighted alternative is a model that quietly never predicts the class at all. The trade-off should be stated, not hidden behind macro-F1.

## Learning curve - how many labels were actually needed?

| training rows | category macro-F1 | materiality macro-F1 | direction macro-F1 | high recall |
|---|---|---|---|---|
| 1,000 | 0.5301 | 0.5303 | 0.5867 | 0.136 |
| 2,500 | 0.7048 | 0.6051 | 0.6895 | 0.404 |
| 5,000 | 0.7601 | 0.6699 | 0.7434 | 0.528 |
| 10,000 | 0.7877 | 0.6823 | 0.7801 | 0.660 |
| 20,000 | 0.8209 | 0.7165 | 0.8076 | 0.711 |
| 26,976 | 0.8292 | 0.7173 | 0.8174 | 0.791 |

From 20,000 to 26,976 rows, category macro-F1 moves +0.0083 while `high`-materiality recall moves +0.081.

**These disagree, and the disagreement is the finding.** The common classes have saturated - the headline metric is flat, and on that evidence alone more labels look wasted. The rarest class is still climbing steeply, because a class at 3.3% of the data only accumulates enough examples to learn late. Judged on macro-F1 the extra labels bought nothing; judged on the class the project exists to predict, they were still paying. A learning curve read on the aggregate alone would have given exactly the wrong answer about when to stop labelling.

## Ablations

| run | category macro-F1 | materiality macro-F1 | high recall | ms/headline |
|---|---|---|---|---|
| FinBERT, three heads | 0.8292 | 0.7173 | 0.791 | 2.46 |
| FinBERT, direction head removed | 0.8383 | 0.7415 | 0.689 | 2.43 |
| DistilBERT, three heads | 0.7983 | 0.6841 | 0.681 | 1.39 |

Dropping the direction head moves category macro-F1 +0.0090 and materiality macro-F1 +0.0242, but `high`-materiality recall -0.102.

**Keep the head.** Removing it makes the summary metrics look slightly better while costing ten points of recall on the class that matters most - the two-head model catches barely two in three high-materiality headlines against four in five. The likely reason is that predicting direction forces the encoder to represent whether news is good or bad, and that representation is what distinguishes a decisive event from a routine one. Chosen on macro-F1 alone, this ablation would have been read as an argument for dropping it.

## Cost and speed

| | per 1,000 headlines |
|---|---|
| teacher (`claude-opus-5`, Batch API) | $1.14, plus minutes of queue latency |
| student on GPU | $0.00, 2.5 s |
| student on CPU | $0.00, ~45 s |

The student needs no API key and no network. That is the project's headline claim, measured rather than asserted.

