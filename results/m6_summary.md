# M6 - confidence: calibrate it, then route on it

M6 was planned as headline against full article. It was reframed (see the project log) to ask where the extra context is worth paying for: leave headlines the student is sure of alone, and send the rest to the teacher. That needs confidence that means something, so calibration comes first. Everything is fitted on the 2018 validation year (5,953 headlines) and judged on the 2019+ test split (8,591), against the teacher's labels.

## Calibration

| head | fitted temperature | test ECE: raw | temperature | temperature + class bias | test accuracy: raw | temperature + class bias |
|---|---|---|---|---|---|---|
| `category` | 1.131 | 0.045 | 0.027 | 0.015 | 0.8611 | 0.8853 |
| `materiality` | 1.316 | 0.027 | 0.016 | 0.010 | 0.8716 | 0.9019 |
| `direction` | 1.303 | 0.049 | 0.013 | 0.007 | 0.8497 | 0.8773 |

**Every head is over-confident** - each fitted temperature is above 1 - and dividing by it cuts calibration error on test without changing a single prediction.

**The class-bias correction raises accuracy, and that is not a free improvement.** It moves the operating point back toward the common classes, undoing what class weighting did on purpose. On `high` materiality, recall falls from 0.791 to 0.383 while precision rises from 0.329 to 0.732. These are two operating points of one model, chosen after training from 2018 data. The higher accuracies are not comparable with M4's baseline, which was class-weighted too and would gain from the same correction.

Before calibration the student's `high` calls carried a mean P(high) of 0.752 at precision 0.329. With the bias correction the calibration error of P(high) itself falls from 0.045 to 0.004.

## Routing: what a little teacher buys

A headline's confidence is its least confident field. Below a cutoff it goes to the teacher's real-time API. Cutoffs were set on 2018 to escalate a target share and applied unchanged to 2019+. Cost is that share times the measured $2.299 per 1,000.

An escalated headline is not scored as certainly right. The teacher, asked again, agrees with its own stored label only 90.8% of the time on all three fields, so escalated rows count at that rate ('adjusted'). That is still optimistic: the headlines the student finds hard are plausibly the ones the teacher is least consistent on. 'Upper' counts them as certain.

**the model as trained (keeps `high` recall)**

| escalated, test | teacher $ per 1,000 | all three fields agree: adjusted | upper | student all three right, on the escalated rows |
|---|---|---|---|---|
| 0.0% | $0.000 | 67.4% | 67.4% | - |
| 4.7% | $0.108 | 70.6% | 71.0% | 23.0% |
| 10.0% | $0.230 | 73.7% | 74.6% | 28.2% |
| 19.9% | $0.458 | 78.7% | 80.5% | 34.3% |
| 32.7% | $0.752 | 84.4% | 87.4% | 38.9% |
| 50.3% | $1.156 | 89.7% | 94.4% | 46.4% |

**with the class-bias correction (higher accuracy, lower `high` recall)**

| escalated, test | teacher $ per 1,000 | all three fields agree: adjusted | upper | student all three right, on the escalated rows |
|---|---|---|---|---|
| 0.0% | $0.000 | 73.5% | 73.5% | - |
| 3.7% | $0.086 | 75.8% | 76.1% | 29.8% |
| 8.8% | $0.202 | 78.6% | 79.4% | 32.9% |
| 18.8% | $0.432 | 83.6% | 85.3% | 37.1% |
| 30.6% | $0.704 | 88.1% | 90.9% | 43.2% |
| 49.9% | $1.148 | 91.9% | 96.5% | 53.9% |

**Confidence finds the student's mistakes.** On the 4.7% of headlines it is least sure of, the student gets all three fields right 23.0% of the time, against 67.4% overall. The 2018 cutoffs escalate almost exactly the intended share in 2019 and 2020, so they transfer across years.

**The teacher's self-consistency is not a hard ceiling.** With the class-bias correction at 50% escalation, adjusted agreement reaches 91.9%, above the teacher's own repeat rate of 90.8%. That is not a leak. Self-consistency is how often two random draws of the teacher agree; a deterministic model that always gave the teacher's most likely answer would match a single stored draw more often than that. The M3 'share of reachable gap' figures, which treat self-consistency as the ceiling, therefore slightly overstate how close the student is.

## Does a calibrated `high` recover the effect on returns? (not yet run)

The gating rule is fixed in `code/m6_downstream.py` from 2018 data alone. Running it needs the licensed price data, so the result is pending; nothing about it is reported here until it has run.

Sources: `m6_calibration.json`. Regenerate with `python code/m6_summary.py`.
