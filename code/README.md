# code/

The distillation pipeline. Populated as milestones land (see the root README status list).

All paths are parameterized (CLI flag / env var) — no absolute paths, no secrets committed.
`--data-dir` / `$DISTILLERY_DATA_DIR` default to `<repo>/data`; `--results-dir` /
`$DISTILLERY_RESULTS_DIR` default to `<repo>/results`. Raw data downloads into `data/raw/`,
which is git-ignored.

## Landed

- `phrasebank.py` — download, parse and split the Financial PhraseBank corpus (Malo et al. 2014).
  Run it alone to fetch the data and print the class balance:
  ```
  python code/phrasebank.py --agreement 50agree
  ```
- `m1_reproduce.py` — M1: the FinBERT sentiment baseline on Financial PhraseBank, in three arms
  evaluated on one shared held-out test split:
  ```
  python code/m1_reproduce.py --arm baseline                                  # TF-IDF + logistic regression
  python code/m1_reproduce.py --arm zeroshot --model ProsusAI/finbert         # off-the-shelf checkpoint
  python code/m1_reproduce.py --arm finetune --model bert-base-uncased        # the actual reproduction
  python code/m1_reproduce.py --arm finetune --model yiyanghkust/finbert-pretrain
  ```
  Each run writes `results/m1_<arm>_<tag>.{txt,json}`.

## Planned

- `teacher_label.py` — run the zero-shot LLM over a public news sample → `{category, materiality, direction}` labels.
- `train_student.py` — fine-tune the FinBERT student on the teacher's labels.
- `evaluate.py` — fidelity (student vs teacher), cost/latency, and the downstream move check.
- `baselines.py` — TF-IDF + logistic regression and DistilBERT comparisons on the distillation task.
