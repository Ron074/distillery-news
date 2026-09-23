# M1 - FinBERT reproduction on Financial PhraseBank

Corpus: Financial PhraseBank `50agree` (Malo et al. 2014). Deterministic stratified split, seed 42: 3874 train / 483 val / 483 test.
All arms scored on the same held-out test set. Hardware: Windows AMD64 python 3.14.5, CPU.

| arm | model | accuracy | macro F1 | weighted F1 | train | inference (ms/sentence) |
|---|---|---|---|---|---|---|
| `baseline` | tfidf(1,2) + logistic regression | 0.8012 | 0.7665 | 0.8007 | 0.5 s | 0.03 |
| `finetune` | yiyanghkust/finbert-pretrain | 0.8696 | 0.8614 | 0.8695 | 71.9 min | 54.21 |
| `finetune` | bert-base-uncased | 0.8675 | 0.8585 | 0.8679 | 73.2 min | 49.90 |
| `zeroshot` | ProsusAI/finbert | 0.8965 | 0.8924 | 0.8975 | n/a | 54.63 |

## Notes

- **ProsusAI/finbert** — Off-the-shelf checkpoint, NOT a clean held-out score: ProsusAI/finbert was itself fine-tuned on Financial PhraseBank, so these test sentences were very likely in its training data. Treat as a pipeline/label-mapping sanity check and an upper reference, not as a reproduction.

## Per-class F1

| model | negative | neutral | positive |
|---|---|---|---|
| tfidf(1,2) + logistic regression | 0.7241 | 0.8547 | 0.7206 |
| yiyanghkust/finbert-pretrain | 0.8793 | 0.8962 | 0.8088 |
| bert-base-uncased | 0.8718 | 0.8951 | 0.8087 |
| ProsusAI/finbert | 0.8992 | 0.9121 | 0.8660 |
