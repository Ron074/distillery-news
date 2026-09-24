"""The label schema the teacher applies and the student learns.

Nine categories x three materiality tiers, plus a direction field that is only committed to when
the text says so outright. The schema is deliberately small: every extra output field is billed at
output-token rates across the whole corpus, and the student only ever learns these three labels.
"""

from __future__ import annotations

CATEGORIES = [
    "merger", "dilution", "clinical", "earnings", "contract",
    "guidance", "legal", "insider", "analyst", "other",
]
MATERIALITY = ["low", "medium", "high"]
DIRECTION = ["bull", "bear", "neutral"]

CATEGORY_DEFINITIONS = """\
merger    - M&A, acquisitions, takeovers, tender offers, mergers of equals, and the termination
            or collapse of any of those.
dilution  - issuance of new shares: offerings, registered directs, ATMs, PIPEs, warrant exercises,
            convertible notes, shelf registrations. Buybacks are NOT dilution; tag them guidance.
clinical  - drug/device trial results, enrollment, endpoints, FDA or other regulatory decisions,
            approvals, rejections, clinical holds, designations.
earnings  - reported financial results for a period: revenue, EPS, margins, quarterly/annual
            results, preliminary results, restatements.
contract  - commercial awards and partnerships: contracts won or lost, purchase orders, licensing,
            distribution and supply agreements, government awards.
guidance  - forward-looking statements BY THE COMPANY, not tied to a reported period: outlook,
            forecasts, raised/lowered/withdrawn guidance, and capital-allocation decisions -
            buybacks, dividend declarations and changes, stock splits and reverse splits.
legal     - litigation, lawsuits, settlements, investigations, subpoenas, enforcement actions,
            regulatory non-compliance, delisting notices, bankruptcy filings, Chapter 11,
            restructuring and going-concern warnings.
insider   - transactions or changes involving insiders and large holders: officer/director buys
            and sells, 13D/13G stakes, activist positions, executive appointments and departures.
analyst   - third-party assessments BY OUTSIDERS, not by the company: broker upgrades and
            downgrades, initiations and reinstatements of coverage, rating changes, price-target
            changes, and published short-seller reports. The distinction from guidance is who is
            speaking - the company itself is guidance, an outside firm is analyst.
other     - anything else, including index roundups, aggregated "biggest movers" lists, reactive
            coverage of a move that has already happened, conference appearances, index additions
            and deletions, product launches, and general market commentary."""

MATERIALITY_DEFINITIONS = """\
high   - a reasonable investor would expect this to move the stock materially on its own:
         going-concern events, major M&A, pivotal trial outcomes, large dilution, guidance
         withdrawal, fraud findings.
medium - plausibly price-relevant but not decisive on its own: in-line results, ordinary
         contracts, routine financings, mid-sized partnerships, leadership changes.
low    - unlikely to move the stock by itself: roundups, reactive "why is X moving" coverage,
         conference attendance, marketing announcements, routine administrative filings."""

DIRECTION_DEFINITIONS = """\
bull    - the text states something a reasonable investor reads as good news for the company.
bear    - the text states something a reasonable investor reads as bad news for the company.
neutral - the text is factual, mixed, or genuinely ambiguous about which way it cuts.

Direction is the field most often gotten wrong by reading the category instead of the words. A
merger headline is not automatically bullish - a terminated merger is bearish. A financing is not
automatically bearish - the words decide. When a headline names an event type but does not say
which way it resolved, the honest answer is neutral."""

SYSTEM_PROMPT = f"""\
You classify financial news for a research dataset. For each item you return exactly three labels:
a category, a materiality tier, and a direction.

CATEGORIES (choose exactly one):
{CATEGORY_DEFINITIONS}

MATERIALITY (choose exactly one):
{MATERIALITY_DEFINITIONS}

DIRECTION (choose exactly one):
{DIRECTION_DEFINITIONS}

Rules:
- Judge only what the given text actually says. Do not use outside knowledge of the company, and
  do not speculate about what the full article might contain.
- Pick the single dominant category. If an item spans several, choose the one a reasonable
  investor would consider the reason the item was published.
- Materiality is about the event's own price relevance, not about how excited the wording is.
  Promotional phrasing does not raise materiality.
- Prefer "other" and "low" over guessing when an item carries no real company-specific event."""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "materiality": {"type": "string", "enum": MATERIALITY},
        "direction": {"type": "string", "enum": DIRECTION},
    },
    "required": ["category", "materiality", "direction"],
    "additionalProperties": False,
}

OUTPUT_CONFIG_FORMAT = {"type": "json_schema", "schema": LABEL_SCHEMA}
