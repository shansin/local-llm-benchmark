# transformation / computed-table

**Prompt:** Below are 28 lines of `id,hours,rate`.

Produce a Markdown table with exactly these columns, in this order:

| ID | Hours | Rate | Pay | Band |

- `Pay` is hours multiplied by rate, rounded to two decimal places.
- `Band` is `low` if Pay is under 500, `mid` if Pay is 500 up to but not including 1500,
  and `high` if Pay is 1500 or more.

Output one row per input line, all 28 of them, in the original order. Output the
table and nothing else — no heading, no explanation, no note about the remaining rows.

DATA
E2001,40,18.85
E2002,18,23.38
E2003,33,47.64
E2004,29,18.25
E2005,23,41.83
E2006,45,37.04
E2007,31,47.49
E2008,41,23.4
E2009,14,60.8
E2010,22,34.24
E2011,29,33.7
E2012,18,20.44
E2013,32,31.26
E2014,42,38.45
E2015,15,36.11
E2016,13,25.5
E2017,36,22.03
E2018,39,31.29
E2019,15,59.02
E2020,32,37.36
E2021,20,24.76
E2022,32,46.43
E2023,44,59.68
E2024,32,33.06
E2025,13,27.41
E2026,33,20.69
E2027,26,59.72
E2028,26,36.01

## Repeat 1 (seed 0)

- Score: 10.0/10 — All 28 rows present in order with correct Pay to two decimals and Band thresholds, table only with no extra prose.
- Tokens/s: 117.2
- Prefill tok/s: 1326.3
- TTFT: 12.30s
- Gen Time: 19.19s
- Output Tokens: 2184
- answer read from `answer_tag`

<details><summary>Reasoning (1734 chars, not scored)</summary>

Let me calculate the Pay (hours × rate) and Band for each row.

E2001: 40 × 18.85 = 754.00 → mid
E2002: 18 × 23.38 = 420.84 → low
E2003: 33 × 47.64 = 1572.12 → high
E2004: 29 × 18.25 = 529.25 → mid
E2005: 23 × 41.83 = 962.09 → mid
E2006: 45 × 37.04 = 1666.80 → high
E2007: 31 × 47.49 = 1472.19 → mid
E2008: 41 × 23.4 = 959.40 → mid
E2009: 14 × 60.8 = 851.20 → mid
E2010: 22 × 34.24 = 753.28 → mid
E2011: 29 × 33.7 = 977.30 → mid
E2012: 18 × 20.44 = 367.92 → low
E2013: 32 × 31.26 = 1000.32 → mid
E2014: 42 × 38.45 = 1614.90 → high
E2015: 15 × 36.11 = 541.65 → mid
E2016: 13 × 25.5 = 331.50 → low
E2017: 36 × 22.03 = 793.08 → mid
E2018: 39 × 31.29 = 1220.31 → mid
E2019: 15 × 59.02 = 885.30 → mid
E2020: 32 × 37.36 = 1195.52 → mid
E2021: 20 × 24.76 = 495.20 → low
E2022: 32 × 46.43 = 1485.76 → mid
E2023: 44 × 59.68 = 2625.92 → high
E2024: 32 × 33.06 = 1057.92 → mid
E2025: 13 × 27.41 = 356.33 → low
E2026: 33 × 20.69 = 682.77 → mid
E2027: 26 × 59.72 = 1552.72 → high
E2028: 26 × 36.01 = 936.26 → mid

Let me double-check a few:
E2001: 40 × 18.85 = 754.00 ✓
E2003: 33 × 47.64 = 33 × 47 + 33 × 0.64 = 1551 + 21.12 = 1572.12 ✓
E2006: 45 × 37.04 = 45 × 37 + 45 × 0.04 = 1665 + 1.8 = 1666.80 ✓
E2007: 31 × 47.49 = 31 × 47 + 31 × 0.49 = 1457 + 15.19 = 1472.19 ✓
E2014: 42 × 38.45 = 42 × 38 + 42 × 0.45 = 1596 + 18.9 = 1614.90 ✓
E2023: 44 × 59.68 = 44 × 59 + 44 × 0.68 = 2596 + 29.92 = 2625.92 ✓
E2027: 26 × 59.72 = 26 × 59 + 26 × 0.72 = 1534 + 18.72 = 1552.72 ✓

Band rules:
- low: Pay < 500
- mid: 500 ≤ Pay < 1500
- high: Pay ≥ 1500

Let me verify E2021: 20 × 24.76 = 495.20 → low (under 500) ✓
E2007: 1472.19 → mid (500 ≤ 1472.19 < 1500) ✓
E2022: 1485.76 → mid ✓
E2027: 1552.72 → high (≥ 1500) ✓

All looks good. Let me format the table.

</details>

**Answer (as scored):**

| ID | Hours | Rate | Pay | Band |
|---|---|---|---|---|
| E2001 | 40 | 18.85 | 754.00 | mid |
| E2002 | 18 | 23.38 | 420.84 | low |
| E2003 | 33 | 47.64 | 1572.12 | high |
| E2004 | 29 | 18.25 | 529.25 | mid |
| E2005 | 23 | 41.83 | 962.09 | mid |
| E2006 | 45 | 37.04 | 1666.80 | high |
| E2007 | 31 | 47.49 | 1472.19 | mid |
| E2008 | 41 | 23.4 | 959.40 | mid |
| E2009 | 14 | 60.8 | 851.20 | mid |
| E2010 | 22 | 34.24 | 753.28 | mid |
| E2011 | 29 | 33.7 | 977.30 | mid |
| E2012 | 18 | 20.44 | 367.92 | low |
| E2013 | 32 | 31.26 | 1000.32 | mid |
| E2014 | 42 | 38.45 | 1614.90 | high |
| E2015 | 15 | 36.11 | 541.65 | mid |
| E2016 | 13 | 25.5 | 331.50 | low |
| E2017 | 36 | 22.03 | 793.08 | mid |
| E2018 | 39 | 31.29 | 1220.31 | mid |
| E2019 | 15 | 59.02 | 885.30 | mid |
| E2020 | 32 | 37.36 | 1195.52 | mid |
| E2021 | 20 | 24.76 | 495.20 | low |
| E2022 | 32 | 46.43 | 1485.76 | mid |
| E2023 | 44 | 59.68 | 2625.92 | high |
| E2024 | 32 | 33.06 | 1057.92 | mid |
| E2025 | 13 | 27.41 | 356.33 | low |
| E2026 | 33 | 20.69 | 682.77 | mid |
| E2027 | 26 | 59.72 | 1552.72 | high |
| E2028 | 26 | 36.01 | 936.26 | mid |

