"""Long-context tasks: documents too long to hold in a summary.

The harness already measures prefill *speed* at 16k tokens. It never measured
whether a model can use what it ingested, which is where models advertising the
same 256k window differ most.

Every document here is deliberately homogeneous: 400 ledger rows that all look
alike, 120 clauses in identical house style. A "needle in a haystack" whose
needle is written in a different voice tests noticing, not retrieval — the
model can find it by style alone. When every line looks like every other line,
the only way to answer is to actually read and aggregate.
"""

from __future__ import annotations

import random

from common import json_checks, write_task

VENDORS = [
    "Northwind Supplies",
    "Cortez Logistics",
    "Ambleside Print",
    "Halloway Instruments",
    "Pemberton Freight",
    "Quillon Software",
    "Rivet & Sons",
    "Saltmarsh Catering",
]
KINDS = ["invoice", "refund", "adjustment"]
STATUSES = ["approved", "rejected", "pending"]
MONTHS = [(1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30)]


def _ledger_rows(rng: random.Random, count: int) -> list[dict]:
    rows = []
    for index in range(count):
        month, days = rng.choice(MONTHS)
        rows.append(
            {
                "id": f"TXN-{1000 + index}",
                "date": f"2025-{month:02d}-{rng.randint(1, days):02d}",
                "vendor": rng.choice(VENDORS),
                "kind": rng.choice(KINDS),
                "amount": round(rng.uniform(45.0, 9800.0), 2),
                "status": rng.choice(STATUSES),
            }
        )
    return rows


def ledger_audit() -> None:
    rng = random.Random(20260822)
    rows = _ledger_rows(rng, 400)
    target = "Halloway Instruments"

    approved_target = [
        r
        for r in rows
        if r["vendor"] == target and r["status"] == "approved" and r["kind"] == "invoice"
    ]
    total = round(sum(r["amount"] for r in approved_target), 2)
    rejected = sum(1 for r in rows if r["status"] == "rejected")
    refunds = [r for r in rows if r["kind"] == "refund" and r["status"] == "approved"]
    largest_refund = max(refunds, key=lambda r: r["amount"])
    pending_vendors = sorted({r["vendor"] for r in rows if r["status"] == "pending"})

    table = "\n".join(
        f"{r['id']} | {r['date']} | {r['vendor']} | {r['kind']} | {r['amount']:.2f} | {r['status']}"
        for r in rows
    )

    prompt = f"""
Below is a transaction ledger with {len(rows)} rows. Each row is:

    ID | DATE | VENDOR | KIND | AMOUNT | STATUS

Answer these five questions about the ledger:

1. What is the total amount of all rows that are vendor "{target}", kind "invoice", and status "approved"? Sum the amounts and give the total to two decimal places.
2. How many rows in the whole ledger have status "rejected"?
3. Of the rows with kind "refund" and status "approved", which has the largest amount? Give its ID.
4. How many distinct vendors appear anywhere in the ledger?
5. How many distinct vendors have at least one row with status "pending"?

Return only a JSON object, with no code fence and no commentary, in exactly this shape:

{{"approved_invoice_total": <number>, "rejected_count": <number>, "largest_refund_id": "<string>", "distinct_vendors": <number>, "vendors_with_pending": <number>}}

LEDGER
{table}
"""

    criteria = f"""
Evaluation criteria:
- The five answers are arithmetic facts about the ledger above; there is exactly one correct
  value for each and no credit for a plausible-looking estimate.
- The task is aggregation over a long homogeneous document, not retrieval of a distinctive
  line. A model that samples part of the ledger and extrapolates will produce a total in
  roughly the right range and still be wrong.
- Output must be a bare JSON object with the five named keys, no code fence, no preamble.
- Do not reward showing the working: the response should contain the JSON and nothing else.

The correct answers are: approved_invoice_total {total:.2f}, rejected_count {rejected},
largest_refund_id {largest_refund["id"]}, distinct_vendors {len(VENDORS)},
vendors_with_pending {len(pending_vendors)}.
"""

    write_task(
        "longcontext",
        "ledger-audit",
        difficulty="hard",
        prompt=prompt,
        criteria=criteria,
        checks=json_checks(
            {
                "rejected_count": rejected,
                "largest_refund_id": largest_refund["id"],
                "distinct_vendors": len(VENDORS),
                "vendors_with_pending": len(pending_vendors),
            }
        )
        # A half-unit tolerance on the sum: the answer is right or wrong by
        # thousands, and failing a correct total over the last cent would be
        # grading the model's rounding rather than its arithmetic.
        + [
            {
                "type": "json_path",
                "path": "approved_invoice_total",
                "equals": total,
                "tolerance": 0.5,
            }
        ],
    )


SUBJECTS = [
    "equipment procurement",
    "travel reimbursement",
    "contractor onboarding",
    "data retention",
    "premises access",
    "external publication",
    "software licensing",
    "incident reporting",
]
ORGANS = [
    "the Operations Committee",
    "the Finance Office",
    "the Compliance Lead",
    "the Directorate",
]


def policy_lookup() -> None:
    """Multi-hop retrieval: the answer is only reachable by following references.

    Clause 63 sets a threshold, clause 11 says that threshold is superseded for
    one category, and clause 88 names the approver for that category. Any single
    clause read in isolation gives a confident wrong answer, which is what makes
    this discriminate rather than merely take up context.
    """
    rng = random.Random(4711)
    clauses = []
    for number in range(1, 121):
        subject = SUBJECTS[number % len(SUBJECTS)]
        organ = ORGANS[number % len(ORGANS)]
        amount = rng.randrange(500, 20000, 250)
        window = rng.choice([5, 10, 14, 21, 30])
        clauses.append(
            f"{number}. Requests concerning {subject} shall be submitted to {organ} not "
            f"later than {window} working days before the commitment date, and shall be "
            f"accompanied by a written justification where the sum committed exceeds "
            f"{amount} units. Records of the determination are retained under clause "
            f"{rng.randint(1, 120)}."
        )

    clauses[62] = (
        "63. Requests concerning equipment procurement are approved by the Operations "
        "Committee where the sum committed is 7500 units or less, and by the Directorate "
        "where it exceeds that figure. This clause is subject to clause 11."
    )
    clauses[10] = (
        "11. Notwithstanding clause 63, equipment classified as calibrated instrumentation "
        "is treated as a restricted category, and the thresholds in clause 63 do not apply "
        "to it. Approval for restricted categories is determined under clause 88."
    )
    clauses[87] = (
        "88. Restricted categories are approved by the Compliance Lead irrespective of the "
        "sum committed, save that any commitment above 40000 units additionally requires "
        "counter-signature by the Directorate."
    )

    body = "\n\n".join(clauses)
    prompt = f"""
The document below is an internal policy with 120 numbered clauses.

A department wishes to purchase a calibrated instrumentation unit — that is, equipment
classified as calibrated instrumentation — committing 12,000 units.

Answer these three questions strictly from the document:

1. Who approves this request? Give the name exactly as the document writes it.
2. Is a counter-signature by the Directorate required for this commitment? Answer true or false.
3. Which clause number states the rule that decides the answer to question 1?

Return only a JSON object, no code fence, no commentary:

{{"approver": "<string>", "countersignature_required": <true|false>, "governing_clause": <number>}}

POLICY
{body}
"""

    criteria = """
Evaluation criteria:
- The answer requires following a chain: clause 63 appears to settle the question by amount,
  clause 11 removes calibrated instrumentation from clause 63's thresholds entirely, and
  clause 88 supplies the actual approver. A model that stops at clause 63 will answer "the
  Directorate" with high confidence — that is the error this task is built to catch.
- 12,000 units is below the 40,000-unit counter-signature threshold in clause 88, so no
  counter-signature is required. A model that carries over the amount-based reasoning from
  clause 63 tends to answer true here.
- The governing clause is 88; clause 11 is the redirection, not the rule.
- Bare JSON with the three named keys, no fence, no working shown.

The correct answers are: approver "the Compliance Lead", countersignature_required false,
governing_clause 88.
"""

    write_task(
        "longcontext",
        "policy-lookup",
        difficulty="hard",
        prompt=prompt,
        criteria=criteria,
        checks=json_checks(
            {
                "approver": "the Compliance Lead",
                "countersignature_required": False,
                "governing_clause": 88,
            }
        ),
    )


def scattered_facts() -> None:
    """Three facts, 200 entries apart, that only mean something together."""
    rng = random.Random(90210)
    names = [
        "Adeyemi",
        "Bratton",
        "Castellanos",
        "Duflot",
        "Eriksen",
        "Farrow",
        "Gulbrandsen",
        "Haruna",
        "Ibarra",
        "Jankowski",
        "Kovac",
        "Lindqvist",
    ]
    entries = []
    for week in range(1, 121):
        lead = names[week % len(names)]
        entries.append(
            f"Week {week}: {lead} chaired. Headcount stood at {rng.randint(40, 90)}. "
            f"{rng.randint(2, 14)} tickets were closed; {rng.randint(0, 5)} were reopened. "
            f"The standing item on tooling was deferred."
        )

    entries[6] = (
        "Week 7: Duflot chaired. Headcount stood at 61. The team agreed that the Halberd "
        "migration would begin in the week following the second quarterly review. 9 tickets "
        "were closed; 1 was reopened."
    )
    entries[48] = (
        "Week 49: Kovac chaired. Headcount stood at 74. It was noted that quarterly reviews "
        "fall in weeks 13, 26, 52 and 78 of the programme calendar. 6 tickets were closed; "
        "2 were reopened."
    )
    entries[38] = (
        "Week 39: Castellanos chaired. Headcount stood at 83. The Halberd migration was "
        "recorded as complete, 12 weeks after it began. 4 tickets were closed; 0 were "
        "reopened."
    )

    body = "\n\n".join(entries)
    prompt = f"""
Below are 120 weekly minutes from a programme's operations meeting.

Using only what the minutes say, work out:

1. In which programme week did the Halberd migration begin?
2. In which programme week did it complete?
3. Who chaired the meeting in the week the migration began?

Return only a JSON object, no code fence, no commentary:

{{"began_week": <number>, "completed_week": <number>, "chair_at_start": "<string>"}}

MINUTES
{body}
"""

    chair_at_start = names[27 % len(names)]
    criteria = f"""
Evaluation criteria:
- No single entry contains the start week. Week 7 says the migration begins the week after
  the second quarterly review; week 49 — 42 entries further down, and *after* the answer in
  document order — says quarterly reviews fall in weeks 13, 26, 52 and 78. The second review
  is therefore week 26 and the migration begins in week 27.
- The completion week is stated plainly in the week 39 entry. Getting that one right while
  missing the start week is the expected partial result, and the per-field checks score it
  as such.
- The chair must be read off the week 27 entry, not the week 7 entry that announced the plan
  nor the week 39 entry that closed it.
- Bare JSON with the three named keys, no fence, no working shown.

The correct answers are: began_week 27, completed_week 39, chair_at_start "{chair_at_start}".
"""

    write_task(
        "longcontext",
        "scattered-facts",
        difficulty="hard",
        prompt=prompt,
        criteria=criteria,
        checks=json_checks(
            {"began_week": 27, "completed_week": 39, "chair_at_start": chair_at_start}
        ),
    )


def build() -> None:
    ledger_audit()
    policy_lookup()
    scattered_facts()
