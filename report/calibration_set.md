# Annotator calibration set

Training material for the hand-labelled gold check. Every headline below is **invented** — none
appears in the corpus, the gold sheet, or any labelled file. They target the specific boundaries
the taxonomy turned out to be soft on, identified from the 18 headlines where two independently
run annotator arms disagreed with each other.

Work through each group, commit to an answer, then read the reasoning. The point is not to
memorise answers but to internalise the rule each group turns on.

## Rule zero: label the headline, nothing else

The teacher was given the headline text and nothing else — no ticker, no date, no article body.
Judge the same way. If a headline is about Philips and the row's ticker is Garmin, that is not
your problem: read the words in front of you.

Three rules from the rubric do most of the work:

1. **Judge only what the text says.** No outside knowledge of the company, no guessing at the body.
2. **Pick the single dominant category** — the reason a reasonable investor would say the item was
   published.
3. **Prefer `other` and `low`** when the item carries no real company-specific event.

---

## Group 1 — an earnings *preview* is not `earnings`

`earnings` means **reported results for a period**. A piece anticipating results reports nothing.

| invented headline | category | materiality |
|---|---|---|
| Northgate Financial Q3 EPS $0.62 Beats $0.55 Estimate, Revenue Up 11% | `earnings` | `medium` |
| Earnings Season Preview: Northgate Financial Near Buy Zone With EPS Due Thursday | `other` | `low` |
| Northgate Financial Reports Next Week: Wall Street Expects Revenue Growth | `other` | `low` |
| Factors Likely To Decide Northgate Financial's Fate In Q3 Earnings | `other` | `low` |

**Why `other` and not `guidance`?** `guidance` is forward-looking statements **by the company**.
A journalist or the street speculating about an upcoming quarter is not the company speaking.

**Why not `analyst`?** `analyst` is a specific list: upgrades, downgrades, initiations,
reinstatements, rating changes, price-target changes, published short-seller reports. "Wall Street
expects growth" is aggregated sentiment in a wire story, not a rating action.

So previews land in `other`, and `low` — nothing has happened yet.

---

## Group 2 — the publisher is not the speaker

Wire services attach their own branding to headlines. It describes who printed the item, not who
acted in it.

| invented headline | category | materiality |
|---|---|---|
| Flowstone Industrial Beats Zacks Estimate - Analyst Blog | `earnings` | `medium` |
| U.S. Rig Count Holds Steady At 1,700 - Analyst Blog | `other` | `low` |
| Cadence Metals Upgraded To Outperform At Raymond James - Analyst Blog | `analyst` | `medium` |

The suffix is identical in all three; the category is driven entirely by the event. Beating an
estimate is reported results. A rig count is industry data. An upgrade is a rating action.

**The test is who performed the action in the sentence**, never who published the sentence.

---

## Group 3 — fund portfolio changes versus a disclosed stake

`insider` covers "officer/director buys and sells, 13D/13G stakes, activist positions, executive
appointments and departures." The question is whether a *specific, disclosed position in this
company* is the news.

| invented headline | category | materiality |
|---|---|---|
| 13D Filing: Brightwater Partners Discloses 7.4% Stake In Camber Industries | `insider` | `medium` |
| Camber Industries Shareholder CTW Group Urges Board To Add Two Independent Directors | `insider` | `medium` |
| Harrowgate Advisors LLC Buys Albemarle, Xerox, Camber Industries, Sells Three Others | `other` | `low` |
| FPA Capital Adds Two Stakes To Portfolio In First Quarter | `other` | `low` |

Rows 1–2 are a disclosed stake and an activist action — both named in the rubric.

Rows 3–4 are quarterly portfolio-disclosure roundups. The subject is the *fund*, not any company
in the list, and the list is incidental. These are `other` and `low`, like any roundup.

**The test:** is the news "someone took a position in this company," or "here is what a fund did
last quarter"?

---

## Group 4 — reactive coverage of a move that already happened

The rubric puts "reactive coverage of a move that has already happened" in `other` explicitly.

| invented headline | category | materiality |
|---|---|---|
| Why Flowstone Industrial Shares Are Trading Higher Today | `other` | `low` |
| Flowstone Industrial Up 4.2% Since Last Earnings Report: Can It Continue? | `other` | `low` |
| Flowstone Industrial (FLS) Catches Eye: Stock Surges 15.2% - Tale Of The Tape | `other` | `low` |

The second one is the trap: it mentions earnings, but the *event* is a price move, and the
earnings report is the reference point rather than the news. Nothing is being reported here.

---

## Group 5 — `guidance` is capital allocation too

`guidance` is broader than its name suggests. It takes forward-looking company statements **and**
capital-allocation decisions.

| invented headline | category | materiality |
|---|---|---|
| Caldera Mining Board Authorizes $500 Million Share Repurchase Program | `guidance` | `medium` |
| Sable Energy Declares Quarterly Dividend Of $0.28, Up From $0.24 | `guidance` | `medium` |
| Novatek Industries Withdraws Full-Year Revenue Guidance Citing Order Delays | `guidance` | `high` |
| Veltrix Pharmaceuticals Announces Closing Of $45 Million Registered Direct Offering | `dilution` | `high` |
| Caldera Mining Announces 1-For-10 Reverse Stock Split | `guidance` | `medium` |

Buybacks, dividends and splits are `guidance`, not `dilution` — **`dilution` is issuance only.**
Guidance withdrawal is `high`: the rubric names it.

---

## Group 6 — restructuring, and headlines about another company

| invented headline | category | materiality |
|---|---|---|
| Philgate NV To Restructure Lumitech Automotive LED Unit - Analyst Blog | `legal` | `medium` |
| Bloomberg: Canada To Miss Deadline For Quick Pipeline Sale | `other` | `low` |
| Ebola Contact List Is Growing | `other` | `low` |

Row 1: the rubric files "restructuring and going-concern warnings" under `legal`. Counter-intuitive
if you read the label name alone, which is exactly why the definitions exist. And note you are
labelling the *headline* — it does not matter that the row's ticker is some other company.

Rows 2–3 carry no company-specific corporate event at all. Government deadlines and public-health
news are `other` and `low`, however dramatic the wording. **Materiality is about the event's own
price relevance, not how urgent it sounds.**

---

## Group 7 — product and service launches

| invented headline | category | materiality |
|---|---|---|
| Verigate Gearing Up To Launch Online TV Streaming Service | `other` | `low` |
| Orbital Dynamics Awarded $210 Million Air Force Satellite Servicing Contract | `contract` | `medium` |
| Orbital Dynamics Signs Multi-Year Supply Agreement With Harrowgate Aerospace | `contract` | `medium` |

The rubric lists "product launches" under `other`. `contract` is for **commercial awards and
agreements with a counterparty** — a contract won, an order, a licence, a supply or distribution
deal. A company launching its own product has no counterparty.

---

## Group 8 — investor days and transcripts

| invented headline | category | materiality |
|---|---|---|
| Veltrix Pharmaceuticals CEO Hosts Investor & Analyst Day 2026 (Transcript) | `other` | `low` |
| Verigate's CEO On Q1 2026 Results - Earnings Call Transcript | `other` | `low` |
| Verigate Reports Q1 Revenue Of $4.1B, EPS $1.12 | `earnings` | `medium` |

"Conference appearances" are `other` in the rubric. A call *transcript* is a publication event, not
a reporting event — the results themselves are the separate `earnings` item. Note that `analyst`
appears in the first headline's title and is irrelevant: it names an audience, not a rating action.

---

## If you remember five things

1. **Headline only.** Ignore ticker and date.
2. **Who acted, not who published.** "Analyst Blog" means nothing.
3. **Reported period = `earnings`. Anticipated period = `other`.**
4. **Buybacks, dividends, splits = `guidance`. Issuance = `dilution`.**
5. **`other` and `low` are correct answers, not giving up.** Roughly half this corpus is roundups
   and reactive coverage.

And `unsure` is always available. A row you genuinely cannot resolve is excluded from scoring, and
which rows were hard is itself worth reporting.
