# Annotator quiz — 22 items

Practice before labelling the gold sheet. **Every headline here is invented.** None is drawn from
`data/gold/gold_blank_pilot2.csv` or from any corpus file, so working through this cannot
contaminate the sheet you are about to label.

How to use it: write your 22 answers down somewhere before scrolling to the key. The score is not
the point — the point is to find which distinctions you do not have yet. The answer key names the
rule that decides each one, so a miss tells you which line of the rubric to re-read.

Rubric: `code/taxonomy.py` (`SYSTEM_PROMPT`), or the copy in `data/gold/labeling_key.txt`.
Canonical examples of each category: `report/calibration_set.md`.

---

## Part A — category (14 items)

Give one of: `merger` `dilution` `clinical` `earnings` `contract` `guidance` `legal` `insider`
`analyst` `other`

1. Northwind Biosciences Announces $45 Million Underwritten Public Offering of Common Stock
2. Pinnacle Robotics Board Authorizes $200 Million Share Repurchase Program
3. Meridian Capital Downgrades Halcyon Energy To Underweight, Cuts Target To $4
4. Halcyon Energy Lowers Full-Year Revenue Outlook To $310-320 Million
5. Sable Creek Resources Terminates Merger Agreement With Fairmont Holdings
6. Thornbury Pharmaceuticals Receives Complete Response Letter From FDA For TBX-401
7. Why Is Castellan Freight Stock Up 18% Today?
8. Castellan Freight Names Dana Reyes As Chief Financial Officer, Effective January 5
9. Ridgeline Defense Awarded $87 Million U.S. Army Logistics Support Contract
10. Varenne Diagnostics Reports Fourth-Quarter Revenue Of $41.2 Million, EPS Of $(0.18)
11. Securities And Exchange Commission Opens Formal Investigation Into Kestrel Mining Disclosures
12. Short-Seller Grayvale Research Publishes Report Alleging Overstated Reserves At Kestrel Mining
13. Orbis Materials To Join S&P SmallCap 600 Effective April 1
14. Larkspur Therapeutics Files $150 Million Shelf Registration Statement On Form S-3

## Part B — materiality (4 items)

Give one of: `low` `medium` `high`

15. Astra Vale Holdings Issues Going-Concern Warning, Says Cash Insufficient Through Next Quarter
16. Thornbury Pharmaceuticals To Present At The Brightwater Healthcare Conference On March 12
17. Ridgeline Defense Signs Three-Year Supply Agreement With Regional Transit Authority
18. Pinnacle Robotics Announces Groundbreaking, Revolutionary New Warehouse Automation Platform

## Part C — direction (4 items)

Give one of: `bull` `bear` `neutral`

19. Sable Creek Resources Terminates Merger Agreement With Fairmont Holdings
20. Northwind Biosciences Announces Pricing Of $45 Million Offering At $2.10 Per Share, A 22% Discount To Last Close
21. Varenne Diagnostics Announces Topline Results From Phase 3 VANTAGE Trial
22. Halcyon Energy Raises Full-Year EPS Guidance To $1.40-1.50 From $1.20-1.30

---
---

## Answer key

Do not read until you have written your 22 answers down.

### Part A — category

| # | answer | the rule that decides it |
|---|---|---|
| 1 | `dilution` | New shares issued. The plain case. |
| 2 | `guidance` | **Buybacks are not dilution.** A repurchase removes shares; it is a capital-allocation decision, which the rubric files under `guidance`. |
| 3 | `analyst` | An outside firm is speaking. |
| 4 | `guidance` | The company is speaking, and it is forward-looking rather than a reported period. **Pairs with 3: the only difference is who speaks.** |
| 5 | `merger` | The rubric puts "the termination or collapse of any of those" inside `merger`. The category is the event type, not the outcome. |
| 6 | `clinical` | An FDA decision. A complete response letter is a rejection, but that affects direction, not category. |
| 7 | `other` | Reactive coverage of a move that already happened. No company event is being announced. |
| 8 | `insider` | `insider` covers executive appointments and departures, not only share transactions. |
| 9 | `contract` | A government award. |
| 10 | `earnings` | Reported results for a period. |
| 11 | `legal` | An investigation. |
| 12 | `analyst` | **A published short-seller report is named explicitly under `analyst`.** The allegation is not itself an enforcement action, so it is not `legal` — compare 11, where the regulator acts. |
| 13 | `other` | Index additions and deletions are listed under `other`. |
| 14 | `dilution` | Shelf registrations are named in the `dilution` definition, even though no shares have been sold yet. |

### Part B — materiality

| # | answer | the rule that decides it |
|---|---|---|
| 15 | `high` | Going-concern events are named in the `high` definition. |
| 16 | `low` | Conference attendance is named in the `low` definition. |
| 17 | `medium` | An ordinary contract: plausibly price-relevant, not decisive on its own. |
| 18 | `low` | **"Promotional phrasing does not raise materiality."** Strip "groundbreaking, revolutionary" and it is a product launch, which the rubric files as `other` / `low`. |

### Part C — direction

| # | answer | the rule that decides it |
|---|---|---|
| 19 | `bear` | Same headline as 5, and the two fields disagree: category `merger`, direction `bear`. **A merger headline is not automatically bullish.** |
| 20 | `bear` | A financing is not automatically bearish either — here the words do say it: priced at a 22% discount. Compare 1, the same offering without a price, which is `neutral`. |
| 21 | `neutral` | The headline names the event but never says how it resolved. **"When a headline names an event type but does not say which way it resolved, the honest answer is neutral."** |
| 22 | `bull` | Guidance raised, stated outright. |

---

## Reading your result

Four traps are deliberate, and they are the four that cost the most on a real sheet:

- **2 and 14** — buyback vs shelf. Both are share-count events; one is `guidance` and one is
  `dilution`.
- **3 vs 4** — who is speaking. Identical subject matter, different category.
- **11 vs 12** — a regulator acting is `legal`; an outside firm publishing is `analyst`.
- **5 vs 19** — category and direction are independent fields. Reading direction off the category
  is, per the rubric, the single most common error.

If you missed any of those four pairs, re-read that definition before opening the sheet — those
are the ones that recur. A miss on 7, 13 or 18 means `other` and `low` are being under-used; the
rubric says to prefer them over guessing when no real company-specific event is present.

Rows you genuinely cannot place go in as `unsure`, not as a guess. They are excluded from scoring,
and which rows are hard is itself a reportable finding.
