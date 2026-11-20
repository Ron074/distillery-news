# M3 checkpoint, verified and re-measured on a laptop CPU

The saved student (`stat_finbert_3head_2026-11-12`, sha256 `18464ee0de3e8735...`) is checked before anything is measured: sha256, label sets, prompt sha1, split counts, test scores equal m3_3head.json, test uids equal preds_3head.csv. All passed.

## It is the M3 model, on CPU as on GPU

| head | accuracy, this CPU | accuracy, M3 (T4) | labels differing from the T4 run | mean confidence | ECE (10 bins) |
|---|---|---|---|---|---|
| `category` | 0.8611 | 0.8611 | 0 of 8,591 | 0.903 | 0.045 |
| `materiality` | 0.8716 | 0.8716 | 0 of 8,591 | 0.898 | 0.027 |
| `direction` | 0.8497 | 0.8497 | 0 of 8,591 | 0.899 | 0.049 |

Not one of the 8,591 test labels changes between the GPU that trained the model and this CPU, so the demo runs the model the report describes.

## Confidence, and what it is worth

Confidence is the top softmax probability of a head; a headline's confidence is that of its *least* confident field, since one shaky field is enough to want a second opinion.

| cutoff | share of headlines below it | any-field error rate below | at or above |
|---|---|---|---|
| 0.5 | 5.3% | 76.9% | 30.1% |
| 0.6 | 19.9% | 64.9% | 24.6% |
| 0.7 | 33.3% | 60.4% | 18.7% |
| 0.8 | 46.7% | 55.4% | 12.6% |
| 0.9 | 61.1% | 48.0% | 8.4% |

Error here means disagreeing with the teacher on at least one of the three fields. Low confidence picks out the headlines the student gets wrong: below 0.5 they are wrong on some field three times in four, against under a third above it. That is what confidence routing (M6) relies on, and it holds.

**Calibration: mildly over-confident overall, badly over-confident on `high`.** Every head's mean confidence exceeds its accuracy, by 2.7 to 4.9 points. The rare class is different: when the student calls a headline `high` it gives that call an average probability of 0.752, and is right 32.9% of the time. The likely cause is the class weighting that made the model find the rare class at all - it raises `high` scores across the board - but that is a hypothesis this run does not test. Either way the raw `high` probabilities cannot be read as probabilities, so any confidence gate on `high` should be set on calibrated scores (temperature or per-class scaling fitted on the 2018 validation year).

## Speed on this laptop

70.4 ms per headline in batches of 32, 85.1 ms one at a time (n = 200), on 4 CPU threads with no GPU - about 12 headlines a second for $0. The September GPU figures (2.3-2.5 ms, a Colab T4) are a different device and are labelled as such wherever they appear.

Source: `m3_checkpoint_check.json` (2026-11-20T02:58:40+00:00, Windows AMD64 python 3.14.5, torch 2.14.0+cpu). Per-headline probabilities: `preds_3head_probs.csv` (git-ignored; regenerate with `code/student_check.py`).
