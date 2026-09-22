# code/

The distillation pipeline. Populated as milestones land (see the root README status list):

- `teacher_label.py` — run the zero-shot LLM over a public news sample → `{category, materiality, direction}` labels.
- `train_student.py` — fine-tune the FinBERT student on the teacher's labels.
- `evaluate.py` — fidelity (student vs teacher), cost/latency, and the downstream move check.
- `baselines.py` — TF-IDF + logistic regression and DistilBERT comparisons.

All paths are parameterized (config / CLI / env-var) — no absolute paths, no secrets committed.
