# M6 Reference Cases — When Full-Article Context Helps, Hurts, or Doesn't Matter

Three real financial-news examples (public press releases + public intraday prices) that map the central
M6 question: **does giving a news classifier the full article body improve its read over the headline alone?**
The honest answer is *it depends* — and these cases illustrate the three regimes. Each uses only public
information (ticker, the press release, and the realized price move).

## Case 1 — Body **rescues** direction (a merger *termination*)
**TOMZ, 2026-09-21.** Headline: *"…Mutually Agree…"* A headline-only classifier tags the story `merger` and,
using that category's average behavior, leans **bullish**. The body reveals the parties agreed to **terminate**
the deal — and the stock fell. The polarity the headline hides is recoverable only from the body.
→ **Full-article helps.** (Same trap: a *failed* clinical trial tagged `clinical`, a *dilutive* offering vs. a
buyback, a contract *loss* vs. a win.)

## Case 2 — Body **amplifies** promotional bias (restated *old* figures)
**GRML, 2026-09-21.** PR: *"…More Than Doubles Control Position in the Sarfartoq Rare Earth District…"* A
classifier reading the **body** *upgrades* materiality and turns bullish — seduced by large numbers ("$2.05B
NPV", "~34% of non-China NdPr") that are actually **restated pre-existing** estimates for a *different* asset
(the ST1 deposit, effective months earlier), plus a geopolitics hook (a national security agreement) and
promotional framing (distributed via a paid investor-relations firm). The **actual new event** was a license
*application* for exploration-only ground — the PR's own text: *"No Mineral Resource has been estimated…
exploration opportunity only… subject to government approval."*
→ **Full-article can make a naïve model *more* bullish on a promoted PR, not less.** Lesson: a robust body model
must **distinguish new figures from restated ones** and **discount speculative/"exploration-only" hedges.**

## Case 3 — Body correctly stays **low**, the move happens anyway (immaterial catalyst)
**TNMG, 2026-09-18.** PR: a subscription-newsletter launch with **~US$14K** of crowdfunding. Both headline and
body classify **low / neutral** — correctly: financially immaterial, and the "AI"/geopolitics language is
descriptive boilerplate with no concrete financials to latch onto. The stock still ran **+155%** (a pre-market
spike that had already faded by the regular-session open).
→ **Full-article correctly declines to amplify** — but low-materiality micro-caps produce large, **unpredictable
tails** driven by float/liquidity/momentum, not content. Lesson: there is a hard ceiling on what any
content-based materiality model can predict for thin-float names; the model's job is **calibration** ("~1-in-8
chance of a ≥15% move"), not calling *which* tail fires.

## Takeaway for M6
The body **helps** when it corrects polarity (Case 1), **hurts** when it amplifies promotional framing built on
concrete-but-stale figures (Case 2), and is **neutral-but-insufficient** when the move is liquidity/momentum-
driven rather than content-driven (Case 3). So a useful full-article model needs to (a) extract
**polarity/subtype**, (b) separate **new vs. restated** figures, and (c) acknowledge that content materiality is
only *one* driver of price — especially for micro-caps. A clean ablation for the report: **headline-only vs.
body model, scored on a *polarity/direction* sub-task over the ambiguous subset**, not just category accuracy.
