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

## Extension under study — headline vs. full article
A headline often hides or distorts meaning. We test whether feeding the model the **full article body** helps.
It cuts three ways — see [`report/m6_reference_cases.md`](report/m6_reference_cases.md) for real examples where
the body *rescues* direction, *amplifies* promotional bias, and *correctly stays neutral* while the stock moves
anyway. The framing question: **does more context improve impact prediction, and on which cases?**

## Data
- **Financial PhraseBank** — for reproducing the FinBERT sentiment baseline (public).
- **FNSPID** — the distillation corpus: public financial-news headlines + tickers + timestamps.
- **Public academic price data** — *downstream-validation only, not training data;* used to turn each headline
  into a realized move. Academic-licensed → kept out of the repo (see [`data/README.md`](data/README.md)).

## Models
- **Teacher:** an existing zero-shot LLM classifier.
- **Student:** FinBERT (fine-tuned); **DistilBERT** as a smaller/faster comparison.
- **Baseline:** TF-IDF + logistic regression (to show the transformer earns its complexity).

## Evaluation
1. **Fidelity** — student-vs-teacher agreement (accuracy / macro-F1, per-class confusion).
2. **Cost & latency** — dollars and milliseconds per 1k headlines, LLM vs. student (the headline result).
3. **Downstream usefulness** — do the student's labels predict real moves as well as the teacher's, controlling
   for company size?
4. **Ablations** — learning curve (labels needed), model size (FinBERT vs DistilBERT), and **headline-only vs.
   full-article** (the extension above).

## Status
Early-stage. This repo is being built as incremental commits; **results are not yet populated** — the sections
below fill in as milestones complete. No results are reported until the corresponding step is done.

- [ ] M1 — Reproduce FinBERT on Financial PhraseBank
- [ ] M2 — Teacher-label a public news sample
- [ ] M3 — Distill the FinBERT student; measure fidelity
- [ ] M4 — Cost / latency vs. baselines
- [ ] M5 — Downstream check against real moves
- [ ] M6 — Full-article vs. headline ablation
- [ ] M7 — Poster, report, repo polish

## Reproduction
*(Will be filled in as `code/` lands. Python 3.11+; CPU for prep, free Colab GPU for training. Keys, if any, are
read from an external `.env` and never committed.)*

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
