# results/

Generated tables and figures, populated **as milestones complete** — nothing is reported here until the
corresponding step is done (no placeholder or fabricated numbers). Expected artifacts:

- `fidelity.txt` — student-vs-teacher agreement (accuracy / macro-F1, confusion) — M3
- `cost_latency.txt` — $ and ms per 1k headlines, LLM vs student vs baselines — M4
- `downstream.txt` — do the student's labels predict real moves as well as the teacher's — M5
- `ablations.txt` — learning curve, FinBERT vs DistilBERT, headline vs full-article — M3/M6
