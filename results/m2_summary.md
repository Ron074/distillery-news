# M2 - teacher-labelled news corpus

**43,296 headlines** labelled by `claude-opus-5` at low reasoning effort, via the Batch API, plus 500 repeated headlines used to measure teacher self-consistency.
Source: FNSPID, 2009-2020, 5,223 tickers.

Labels are not in this repository: they are regenerable from the sample and the code, and the raw corpus is large. See `code/teacher_label.py`.

### Category

| value | n | share |
|---|---|---|
| `other` | 20,650 | 47.7% |
| `analyst` | 7,538 | 17.4% |
| `earnings` | 6,322 | 14.6% |
| `guidance` | 2,063 | 4.8% |
| `insider` | 1,901 | 4.4% |
| `merger` | 1,891 | 4.4% |
| `contract` | 1,105 | 2.6% |
| `legal` | 685 | 1.6% |
| `clinical` | 660 | 1.5% |
| `dilution` | 481 | 1.1% |

### Materiality

| value | n | share |
|---|---|---|
| `low` | 29,469 | 68.1% |
| `medium` | 12,394 | 28.6% |
| `high` | 1,433 | 3.3% |

### Direction

| value | n | share |
|---|---|---|
| `neutral` | 27,162 | 62.7% |
| `bull` | 10,175 | 23.5% |
| `bear` | 5,959 | 13.8% |

### Corpus shape

| property | value |
|---|---|
| headlines | 43,296 |
| distinct tickers | 5,223 |
| median headlines per ticker | 5 |
| exact duplicate titles | 862 (2.0%) |
| duplicates after masking numbers | 1,776 (4.1%) |

### Planned split

Time-based, so the test period is one the model never saw: train on years before 2019 (34,455 rows), hold out 2019 onward (8,841 rows, 20%).

The held-out period contains 235 high-materiality headlines, which is the scarcest thing being measured and so the binding constraint on how precisely recall can be reported.

### External validation of the teacher

Fidelity only shows whether a student copies its teacher; it cannot catch a teacher that is confidently wrong. This check tests a label against a record kept elsewhere.

Of headlines tagged `earnings`, **68.4%** fall within 2 days of a real quarterly report date, against **13.8%** for everything else - a **4.97x lift** over the base rate, across 5,189 tagged headlines.

Reported as lift rather than a raw hit rate: earnings news is common enough here that a bare percentage would look convincing without demonstrating anything.

