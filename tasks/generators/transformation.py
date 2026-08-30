"""Bulk exact transformation: do the same small thing thirty times without drifting.

These tasks have no difficulty at all in the first three records. The failure
mode they catch is laziness — a model that writes fifteen rows and then
"... (remaining records follow the same pattern)", or that silently
abbreviates, or whose field values start drifting once the novelty wears off.
That behaviour varies enormously between models of the same size and is
invisible to every check the suite had before, because a truncated-but-pretty
answer passes `contains_all` and reads well to a judge.
"""

from __future__ import annotations

import random

from common import json_checks, write_task

FIRST = ["Marla", "Devi", "Osk", "Pilar", "Toma", "Ines", "Ruben", "Yuki", "Ada", "Cyril"]
LAST = ["Okonjo", "Vasquez", "Lindgren", "Baptiste", "Novak", "Tanaka", "Mbeki", "Ferrers"]
DEPTS = ["Fabrication", "Quality", "Logistics", "Design", "Field Service"]
SITES = ["Bellweather", "Corran", "Dunmore", "Eastgate"]


def _staff(rng: random.Random, count: int) -> list[dict]:
    return [
        {
            "id": f"E{2001 + i}",
            "first": rng.choice(FIRST),
            "last": rng.choice(LAST),
            "dept": rng.choice(DEPTS),
            "site": rng.choice(SITES),
            "hours": rng.randrange(12, 46),
            "rate": round(rng.uniform(18.0, 62.0), 2),
        }
        for i in range(count)
    ]


def csv_to_json() -> None:
    rng = random.Random(606)
    rows = _staff(rng, 30)
    header = "id,first,last,dept,site,hours,rate"
    csv = "\n".join(
        f"{r['id']},{r['first']},{r['last']},{r['dept']},{r['site']},{r['hours']},{r['rate']}"
        for r in rows
    )

    prompt = f"""
Convert the CSV below into a JSON array. Every row becomes one object with these keys:

- `id` — string, unchanged
- `name` — string, the first and last name joined with a single space
- `dept` — string, unchanged
- `pay` — number, hours multiplied by rate, rounded to two decimal places

Drop the `site` column entirely. Keep the rows in their original order. Convert every row
— all {len(rows)} of them. Do not abbreviate, do not summarise, do not write a placeholder
for the remaining rows.

Return only the JSON array, with no code fence and no commentary.

CSV
{header}
{csv}
"""

    def pay(record: dict) -> float:
        return round(record["hours"] * record["rate"], 2)

    # Spot-check the ends and the middle: a model that abbreviates gets the
    # first right and the last wrong, which is exactly the distinction here.
    expected = {
        "[0].id": rows[0]["id"],
        "[0].name": f"{rows[0]['first']} {rows[0]['last']}",
        "[14].dept": rows[14]["dept"],
        "[29].id": rows[29]["id"],
        "[29].name": f"{rows[29]['first']} {rows[29]['last']}",
    }

    criteria = f"""
Evaluation criteria:
- All {len(rows)} records must be present, in order, with no ellipsis, no "and so on", no
  placeholder object, and no note explaining that the rest follow the same pattern. An
  abbreviated answer is a failed answer however neatly it is formatted.
- `name` is first and last joined by one space. `pay` is hours x rate to two decimals.
  `site` must not appear anywhere in the output.
- Bare JSON array, no code fence, no preamble.
- Correctness of the arithmetic matters more than formatting, but both are required.

Record 1 is {rows[0]["id"]} and record {len(rows)} is {rows[29]["id"]}; a response whose
last record is not {rows[29]["id"]} has dropped data.
"""

    write_task(
        "transformation",
        "csv-to-json",
        difficulty="medium",
        prompt=prompt,
        criteria=criteria,
        checks=json_checks(expected)
        + [
            # Counting the `id` keys is what catches an answer that stops early:
            # every spot-check can pass on a document that is missing its middle.
            {"type": "match_count", "pattern": r'"id"\s*:', "equals": len(rows), "weight": 2.0},
            {"type": "regex", "pattern": r"site", "negate": True},
            {"type": "json_path", "path": "[0].pay", "equals": pay(rows[0]), "tolerance": 0.02},
            {"type": "json_path", "path": "[29].pay", "equals": pay(rows[29]), "tolerance": 0.02},
        ],
    )


def schema_migration() -> None:
    """Restructure rather than reformat: the output shape is not the input shape."""
    rng = random.Random(808)
    rows = _staff(rng, 24)
    by_dept: dict[str, list[dict]] = {}
    for record in rows:
        by_dept.setdefault(record["dept"], []).append(record)

    listing = "\n".join(
        f"{r['id']} | {r['first']} {r['last']} | {r['dept']} | {r['site']} | {r['hours']}h"
        for r in rows
    )
    departments = sorted(by_dept)
    busiest = max(by_dept, key=lambda name: sum(m["hours"] for m in by_dept[name]))

    prompt = f"""
Below are {len(rows)} staff records, one per line:

    ID | NAME | DEPARTMENT | SITE | HOURS

Reorganise them into a JSON object keyed by department. Each department maps to an object
with:

- `headcount` — number of staff in that department
- `total_hours` — the sum of their hours
- `ids` — an array of their IDs, in the order they appear in the input

Then add one further top-level key, `busiest`, whose value is the name of the department
with the greatest total hours.

Return only the JSON object, no code fence, no commentary. The shape is:

{{"<Department>": {{"headcount": <number>, "total_hours": <number>, "ids": ["<string>", ...]}}, ..., "busiest": "<string>"}}

RECORDS
{listing}
"""

    criteria = f"""
Evaluation criteria:
- This is a regrouping, not a reformatting: the output has one entry per department, not
  one per person, so a model that transcribes the input into JSON has not done the task.
- Every department present in the input must appear as a key. There are {len(departments)}:
  {", ".join(departments)}.
- `headcount`, `total_hours` and `ids` must agree with each other — a headcount that does
  not match the length of `ids` is a self-inconsistent answer.
- `busiest` compares total hours, not headcount. The two do not have the same answer here,
  and choosing the largest department instead is the expected error.
- Bare JSON, no fence, no preamble.
"""

    expected: dict[str, object] = {"busiest": busiest}
    for name, members in by_dept.items():
        expected[f"{name}.headcount"] = len(members)
        expected[f"{name}.total_hours"] = sum(m["hours"] for m in members)
        expected[f"{name}.ids[0]"] = members[0]["id"]

    write_task(
        "transformation",
        "schema-migration",
        difficulty="hard",
        prompt=prompt,
        criteria=criteria,
        checks=json_checks(expected)
        + [{"type": "regex", "pattern": r"\bBellweather\b|\bCorran\b", "negate": True}],
    )


def computed_table() -> None:
    """A fixed-width output where the row count is the whole point."""
    rng = random.Random(112358)
    rows = _staff(rng, 28)

    listing = "\n".join(f"{r['id']},{r['hours']},{r['rate']}" for r in rows)
    prompt = f"""
Below are {len(rows)} lines of `id,hours,rate`.

Produce a Markdown table with exactly these columns, in this order:

| ID | Hours | Rate | Pay | Band |

- `Pay` is hours multiplied by rate, rounded to two decimal places.
- `Band` is `low` if Pay is under 500, `mid` if Pay is 500 up to but not including 1500,
  and `high` if Pay is 1500 or more.

Output one row per input line, all {len(rows)} of them, in the original order. Output the
table and nothing else — no heading, no explanation, no note about the remaining rows.

DATA
{listing}
"""

    def band(record: dict) -> str:
        pay = round(record["hours"] * record["rate"], 2)
        return "low" if pay < 500 else ("mid" if pay < 1500 else "high")

    counts = {name: sum(1 for r in rows if band(r) == name) for name in ("low", "mid", "high")}
    criteria = f"""
Evaluation criteria:
- Exactly {len(rows)} data rows, in the original order, plus the header and separator rows.
  A table that stops early or ends with an ellipsis row has failed regardless of how correct
  the rows it did write are.
- `Pay` must be hours x rate to two decimals, and `Band` must follow the stated thresholds.
  The boundaries are inclusive-below: 500 is `mid`, 1500 is `high`.
- No prose before or after the table.
- For reference the correct band distribution is {counts["low"]} low, {counts["mid"]} mid,
  {counts["high"]} high.
"""

    write_task(
        "transformation",
        "computed-table",
        difficulty="medium",
        prompt=prompt,
        criteria=criteria,
        checks=[
            # Header and separator are rows too, hence the +2.
            {
                "type": "line_count",
                "pattern": r"^\s*\|",
                "equals": len(rows) + 2,
                "weight": 2.0,
            },
            {"type": "match_count", "pattern": r"\bE2\d{3}\b", "equals": len(rows)},
            {"type": "match_count", "pattern": r"(?i)\blow\b", "equals": counts["low"]},
            {"type": "match_count", "pattern": r"(?i)\bmid\b", "equals": counts["mid"]},
            {"type": "match_count", "pattern": r"(?i)\bhigh\b", "equals": counts["high"]},
        ],
    )


def build() -> None:
    csv_to_json()
    schema_migration()
    computed_table()
