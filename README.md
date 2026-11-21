# The Distillery! : News

*Distilling an expensive LLM into a small, free model that reads and evaluates financial news.*

**STAT UN3108: Applied Deep Learning and AI (Columbia University) — final project.**

---

## What this is
An expensive large language model (LLM) can read a financial-news item and judge it — what *kind* of news it is
and how *market-moving* it is likely to be — but doing that costs money per call and needs an API key. **This
project trains a small model we own to do the same job for free, locally, by distillation:** the LLM (the
*teacher*) labels a large set of public news; a compact model (the *student*, based on FinBERT) learns to
reproduce those labels. The deliverable is the **student model** and an honest measurement of how well it
matches the teacher.

> **Research question:** *Can a small, fine-tuned model match a large zero-shot LLM at reading and evaluating
> financial news — on accuracy, cost, and a real-world downstream check?*

## Why it's interesting
- **Real deep learning** — fine-tuning a transformer (the course's core), not just calling an API.
- **A free, offline classifier** — once distilled, the student runs with no API key and no per-call cost.
- **A transferable takeaway** — "distill what you pay for into something you own" is a pattern classmates can
  reuse for their own domain.

## Approach
```
  Public news  ──►  Zero-shot LLM (TEACHER, $/call)  ──►  labels: {category, materiality, direction}
                                                                 │
                                              fine-tune FinBERT (STUDENT, free/local)
                                                                 │
              ┌──────────────────┬──────────────────────────────┴───────────────┐
              ▼                   ▼                                              ▼
          FIDELITY           COST / LATENCY                         DOWNSTREAM CHECK
      student vs teacher    $ & ms per 1k headlines,     do the student's labels line up with real
      agreement (acc/F1)    LLM vs student               price moves as well as the teacher's?
```

## Extension — where is extra context worth paying for?
A headline often hides or distorts meaning; see [`report/m6_reference_cases.md`](report/m6_reference_cases.md)
for real examples where the article body *rescues* direction, *amplifies* promotional bias, or *correctly stays
neutral* while the stock moves anyway. The plan was to compare headline-only labels against full-article labels.
It became **confidence routing** instead: calibrate the student, leave the headlines it is sure of alone, and
send only the uncertain ones to the teacher. The body comparison itself was not run, because none of the
48,040 sampled headlines carries article text in the corpus file used. It stays as future work (below).

## Data
- **Financial PhraseBank** — for reproducing the FinBERT sentiment baseline (public).
- **FNSPID** — the distillation corpus: public financial-news headlines + tickers + timestamps.
- **Public academic price data** — *downstream-validation only, not training data;* used to turn each headline
  into a realized move. Academic-licensed → kept out of the repo (see [`data/README.md`](data/README.md)).

## Models
- **Teacher:** an existing zero-shot LLM classifier.
- **Student:** FinBERT (fine-tuned); **DistilBERT** as a smaller/faster comparison.
- **Baseline:** TF-IDF + logistic regression on the same split. It nearly matches the student on category
  and overall materiality accuracy; the student earns its cost on direction and the rare `high` tier (M4).

## Evaluation
1. **Fidelity** — student-vs-teacher agreement (accuracy / macro-F1, per-class confusion).
2. **Cost & latency** — dollars and milliseconds per 1k headlines, LLM vs. student (the headline result).
3. **Downstream usefulness** — do the student's labels predict real moves as well as the teacher's, controlling
   for company size?
4. **Ablations** — learning curve (labels needed), model size (FinBERT vs DistilBERT), and two-head vs three-head.
5. **Confidence** — calibration fitted on 2018, and routing the least-confident headlines to the teacher
   (the extension above).

## Status
Being built as incremental commits. M1 to M6 are complete and their results are in
[`results/`](results/); the poster and report (M7) are in progress. No result is reported until the
corresponding step has actually been run.

- [x] M1 — Reproduce FinBERT on Financial PhraseBank → [`results/m1_summary.md`](results/m1_summary.md)
- [x] M2 — Teacher-label a public news sample → [`results/m2_summary.md`](results/m2_summary.md)
- [x] M3 — Distill the FinBERT student; measure fidelity → [`results/m3_summary.md`](results/m3_summary.md)
- [x] M4 — Cost / latency vs. baselines → [`results/m4_summary.md`](results/m4_summary.md)
- [x] M5 — Downstream check against real moves → [`results/m5_summary.md`](results/m5_summary.md)
- [x] M6 — Confidence: calibration and routing, headline only → [`results/m6_summary.md`](results/m6_summary.md)
  *(planned as headline vs. full article; no sampled headline has article text, so the body comparison is
  future work)*
- [ ] M7 — Poster, report, repo polish

## Reproduction
Python 3.11+; CPU for prep, free Colab GPU for training. Keys, if any, are read from an external `.env` and
never committed. Raw data downloads to `data/raw/` on first run and is git-ignored.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only box
.venv/Scripts/python -m pip install -r requirements.txt

python code/phrasebank.py                        # fetch Financial PhraseBank, print class balance
python code/m1_reproduce.py --arm baseline       # TF-IDF + logistic regression
python code/m1_reproduce.py --arm finetune --model bert-base-uncased
python code/m1_reproduce.py --arm finetune --model yiyanghkust/finbert-pretrain
python code/m1_summary.py                        # rebuild results/m1_summary.md
```

After the first run everything is cached locally and M1 needs no network.
The two fine-tunes take ~72 min each on a 4-thread CPU; minutes on a GPU.

M2 calls a paid API, so it reads a key from an external `.env` and always estimates before spending:

```bash
python code/fnspid.py --n 50000                                  # sample FNSPID (downloads ~5.7 GB once)
python code/fnspid.py --filter-sample data/samples/<sample>.csv  # drop template roundups
python code/teacher_label.py estimate --sample data/samples/<clean>.csv   # free: counts tokens only
python code/teacher_label.py submit   --sample data/samples/<clean>.csv --tag full --duplicates 500
python code/teacher_label.py collect  --tag full --wait
python code/m2_summary.py --sample data/samples/<clean>.csv      # rebuild results/m2_summary.md
```

`submit` skips any headline already labelled, so an interrupted or unfunded run resumes rather than
re-paying. `--duplicates` re-asks about N headlines to measure teacher self-consistency, the reference
point for student fidelity. Validation against external records needs WRDS access:

```bash
python code/validate_teacher.py earnings --tag full --sample data/samples/<clean>.csv --wrds-dir <path>
```

### M1 result
Fine-tuned FinBERT reaches **0.870 accuracy / 0.861 macro-F1** on a held-out split of the 50%-agreement
corpus, reproducing the BERT-class numbers Araci (2019) reports, and beating a TF-IDF + logistic-regression
baseline by **9.5 macro-F1 points** — the transformer earns its complexity. Full table, per-class breakdown
and the contamination caveat on the off-the-shelf checkpoint: [`results/m1_summary.md`](results/m1_summary.md).

### M2 result
**43,296 headlines** (2009–2020, 5,223 tickers) labelled by `claude-opus-5` across 10 categories, three
materiality tiers and three directions. Asked the same 500 randomly drawn headlines twice, the teacher gives
the same answer on all three fields **90.8%** of the time, the reference point for any student's fidelity
(a useful one, though not a hard ceiling; see M6). Against outside records,
**68.4%** of `earnings`-tagged headlines fall within two days of a real quarterly report date versus 13.8%
for everything else, a **4.97× lift**. Full distribution and the planned time-based split:
[`results/m2_summary.md`](results/m2_summary.md).

### M3 result
The distilled student reproduces the teacher at **0.861 accuracy on `category`** over a held-out
*time* period (2019–2020) it never trained on. Asked the same headline twice, the teacher repeats its own
category 96.2% of the time; measured against that, the student closes 78% of the gap above a majority-class
baseline. That slightly overstates it, because repeat agreement is not a hard ceiling: a model that always
gave the teacher's most likely answer could exceed it (see [`results/m6_summary.md`](results/m6_summary.md)).
It catches **79% of
high-materiality headlines**, the rarest and most important class. On a laptop CPU with no GPU it
answers in **85 ms per headline for $0**, against **1.9 s and $2.30 per 1,000** for the teacher's
real-time API ($1.135 per 1,000 through the Batch API). On a Colab T4 it takes 2.3 ms. Full cost
table: [`results/m4_summary.md`](results/m4_summary.md).

Verified two ways rather than self-reported: the metrics are recomputable from saved per-headline
predictions (match to four decimals), and a **null test** — training labels permuted so no
headline-label relationship survives — collapses category from 0.861 to **0.116**, near chance for
ten classes. Learning curve, ablations and the precision/recall trade-off:
[`results/m3_summary.md`](results/m3_summary.md).

## Repo layout
```
code/     the distillation pipeline (teacher labeling, student training, evaluation)
data/     how to obtain the public datasets (raw data is git-ignored)
results/  generated tables/figures (populated as milestones complete)
report/   the 2-page IEEE report + supporting notes (incl. m6_reference_cases.md)
poster/   the poster PDF
```

## References
- Araci (2019), *FinBERT.* · Sanh et al. (2019), *DistilBERT.* · Hinton et al. (2015), *Distilling the Knowledge
  in a Neural Network.* · FNSPID dataset. · Financial PhraseBank.

## Acknowledgements
Completed for **STAT UN3108: Applied Deep Learning and AI** (Columbia University). Development assisted by Claude
Code (Anthropic).

### Disclaimer
Research and educational tooling. **Not financial advice.** Any relationship between news labels and price is
descriptive; direction in particular is treated as largely unpredictable.
