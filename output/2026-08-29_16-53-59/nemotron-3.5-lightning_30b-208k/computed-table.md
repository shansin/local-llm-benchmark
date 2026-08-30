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
- Tokens/s: 316.9
- Prefill tok/s: 3140.3
- TTFT: 28.80s
- Gen Time: 31.27s
- Output Tokens: 9837
- answer read from `thinking_field`

<details><summary>Reasoning (16263 chars, not scored)</summary>

Here's a thinking process:

1.  **Understand the Input**: I have 28 lines of data in the format `id,hours,rate`. I need to compute `Pay = hours * rate`, rounded to 2 decimal places. Then determine `Band`: 
   - `low` if Pay < 500
   - `mid` if 500 <= Pay < 1500
   - `high` if Pay >= 1500

2.  **Output Format**: A Markdown table with columns: `ID | Hours | Rate | Pay | Band`. Exactly 28 rows, original order, no heading, no explanation, no notes.

3.  **Plan**: I'll compute each row's Pay and Band, then format the table. I need to be careful with rounding and band boundaries.

Let me list out the data and compute:

Row 1: E2001, 40, 18.85 -> Pay = 40 * 18.85 = 754.00 -> Band: mid (500 <= 754 < 1500)
Row 2: E2002, 18, 23.38 -> Pay = 18 * 23.38 = 420.84 -> Band: low (<500)
Row 3: E2003, 33, 47.64 -> Pay = 33 * 47.64 = 1572.12 -> Band: high (>=1500)
Row 4: E2004, 29, 18.25 -> Pay = 29 * 18.25 = 529.25 -> Band: mid
Row 5: E2005, 23, 41.83 -> Pay = 23 * 41.83 = 962.09 -> Band: mid
Row 6: E2006, 45, 37.04 -> Pay = 45 * 37.04 = 1666.80 -> Band: high
Row 7: E2007, 31, 47.49 -> Pay = 31 * 47.49 = 1472.19 -> Band: mid (since <1500)
Row 8: E2008, 41, 23.4 -> Pay = 41 * 23.4 = 959.40 -> Band: mid
Row 9: E2009, 14, 60.8 -> Pay = 14 * 60.8 = 851.20 -> Band: mid
Row 10: E2010, 22, 34.24 -> Pay = 22 * 34.24 = 753.28 -> Band: mid
Row 11: E2011, 29, 33.7 -> Pay = 29 * 33.7 = 977.30 -> Band: mid
Row 12: E2012, 18, 20.44 -> Pay = 18 * 20.44 = 367.92 -> Band: low
Row 13: E2013, 32, 31.26 -> Pay = 32 * 31.26 = 999.92 -> Band: mid
Row 14: E2014, 42, 38.45 -> Pay = 42 * 38.45 = 1614.90 -> Band: high
Row 15: E2015, 15, 36.11 -> Pay = 15 * 36.11 = 541.65 -> Band: mid
Row 16: E2016, 13, 25.5 -> Pay = 13 * 25.5 = 331.50 -> Band: low
Row 17: E2017, 36, 22.03 -> Pay = 36 * 22.03 = 793.08 -> Band: mid
Row 18: E2018, 39, 31.29 -> Pay = 39 * 31.29 = 1220.31 -> Band: mid
Row 19: E2019, 15, 59.02 -> Pay = 15 * 59.02 = 885.30 -> Band: mid
Row 20: E2020, 32, 37.36 -> Pay = 32 * 37.36 = 1195.52 -> Band: mid
Row 21: E2021, 20, 24.76 -> Pay = 20 * 24.76 = 495.20 -> Band: low (<500)
Row 22: E2022, 32, 46.43 -> Pay = 32 * 46.43 = 1485.76 -> Band: mid (<1500)
Row 23: E2023, 44, 59.68 -> Pay = 44 * 59.68 = 2625.92 -> Band: high
Row 24: E2024, 32, 33.06 -> Pay = 32 * 33.06 = 1057.92 -> Band: mid
Row 25: E2025, 13, 27.41 -> Pay = 13 * 27.41 = 356.33 -> Band: low
Row 26: E2026, 33, 20.69 -> Pay = 33 * 20.69 = 682.77 -> Band: mid
Row 27: E2027, 26, 59.72 -> Pay = 26 * 59.72 = 1552.72 -> Band: high
Row 28: E2028, 26, 36.01 -> Pay = 26 * 36.01 = 936.26 -> Band: mid

Let me double-check all calculations carefully. I'll recompute each one meticulously.

1. E2001: 40 * 18.85 = 754.00 -> mid
2. E2002: 18 * 23.38 = 420.84 -> low
3. E2003: 33 * 47.64 = let's calculate: 47.64 * 33 = 47.64*30 + 47.64*3 = 1429.2 + 142.92 = 1572.12 -> high
4. E2004: 29 * 18.25 = 18.25*29 = 18.25*30 - 18.25 = 547.5 - 18.25 = 529.25 -> mid
5. E2005: 23 * 41.83 = 41.83*20 + 41.83*3 = 836.6 + 125.49 = 962.09 -> mid
6. E2006: 45 * 37.04 = 37.04*45 = 37.04*40 + 37.04*5 = 1481.6 + 185.2 = 1666.80 -> high
7. E2007: 31 * 47.49 = 47.49*31 = 47.49*30 + 47.49 = 1424.7 + 47.49 = 1472.19 -> mid (since <1500)
8. E2008: 41 * 23.4 = 23.4*41 = 23.4*40 + 23.4 = 936 + 23.4 = 959.40 -> mid
9. E2009: 14 * 60.8 = 851.20 -> mid
10. E2010: 22 * 34.24 = 34.24*22 = 34.24*20 + 34.24*2 = 684.8 + 68.48 = 753.28 -> mid
11. E2011: 29 * 33.7 = 33.7*29 = 33.7*30 - 33.7 = 1011 - 33.7 = 977.30 -> mid
12. E2012: 18 * 20.44 = 20.44*18 = 20.44*20 - 20.44*2 = 408.8 - 40.88 = 367.92 -> low
13. E2013: 32 * 31.26 = 31.26*32 = 31.26*30 + 31.26*2 = 937.8 + 62.52 = 1000.32 -> Wait, let me recalculate: 31.26 * 32. 31.26 * 32 = 31.26 * 30 = 937.8, plus 31.26 * 2 = 62.52, total = 1000.32. But earlier I had 999.92. Let me check carefully: 31.26 * 32. 31 * 32 = 992, 0.26 * 32 = 8.32, total = 1000.32. Yes, 1000.32. I'll correct that. -> mid
14. E2014: 42 * 38.45 = 38.45*42 = 38.45*40 + 38.45*2 = 1538 + 76.9 = 1614.90 -> high
15. E2015: 15 * 36.11 = 541.65 -> mid
16. E2016: 13 * 25.5 = 331.50 -> low
17. E2017: 36 * 22.03 = 22.03*36 = 22.03*30 + 22.03*6 = 660.9 + 132.18 = 793.08 -> mid
18. E2018: 39 * 31.29 = 31.29*39 = 31.29*40 - 31.29 = 1251.6 - 31.29 = 1220.31 -> mid
19. E2019: 15 * 59.02 = 885.30 -> mid
20. E2020: 32 * 37.36 = 37.36*32 = 37.36*30 + 37.36*2 = 1120.8 + 74.72 = 1195.52 -> mid
21. E2021: 20 * 24.76 = 495.20 -> low
22. E2022: 32 * 46.43 = 46.43*32 = 46.43*30 + 46.43*2 = 1392.9 + 92.86 = 1485.76 -> mid (since <1500)
23. E2023: 44 * 59.68 = 59.68*44 = 59.68*40 + 59.68*4 = 2387.2 + 238.72 = 2625.92 -> high
24. E2024: 32 * 33.06 = 33.06*32 = 33.06*30 + 33.06*2 = 991.8 + 66.12 = 1057.92 -> mid
25. E2025: 13 * 27.41 = 27.41*13 = 27.41*10 + 27.41*3 = 274.1 + 82.23 = 356.33 -> low
26. E2026: 33 * 20.69 = 20.69*33 = 20.69*30 + 20.69*3 = 620.7 + 62.07 = 682.77 -> mid
27. E2027: 26 * 59.72 = 59.72*26 = 59.72*20 + 59.72*6 = 1194.4 + 358.32 = 1552.72 -> high
28. E2028: 26 * 36.01 = 36.01*26 = 36.01*20 + 36.01*6 = 720.2 + 216.06 = 936.26 -> mid

Let me re-verify the ones I might have miscalculated:
- E2013: 31.26 * 32. 31.26 * 32 = (31 + 0.26)*32 = 992 + 8.32 = 1000.32. Correct.
- E2007: 47.49 * 31 = (47 + 0.49)*31 = 1457 + 15.19 = 1472.19. Correct.
- E2022: 46.43 * 32 = (46 + 0.43)*32 = 1472 + 13.76 = 1485.76. Correct.
- E2021: 24.76 * 20 = 495.20. Correct.
- E2016: 25.5 * 13 = 331.50. Correct.

All Pay values rounded to 2 decimal places. They already are.

Now, Band rules:
- low: Pay < 500
- mid: 500 <= Pay < 1500
- high: Pay >= 1500

Let's assign bands with the corrected Pay:
1. E2001: 754.00 -> mid
2. E2002: 420.84 -> low
3. E2003: 1572.12 -> high
4. E2004: 529.25 -> mid
5. E2005: 962.09 -> mid
6. E2006: 1666.80 -> high
7. E2007: 1472.19 -> mid
8. E2008: 959.40 -> mid
9. E2009: 851.20 -> mid
10. E2010: 753.28 -> mid
11. E2011: 977.30 -> mid
12. E2012: 367.92 -> low
13. E2013: 1000.32 -> mid
14. E2014: 1614.90 -> high
15. E2015: 541.65 -> mid
16. E2016: 331.50 -> low
17. E2017: 793.08 -> mid
18. E2018: 1220.31 -> mid
19. E2019: 885.30 -> mid
20. E2020: 1195.52 -> mid
21. E2021: 495.20 -> low
22. E2022: 1485.76 -> mid
23. E2023: 2625.92 -> high
24. E2024: 1057.92 -> mid
25. E2025: 356.33 -> low
26. E2026: 682.77 -> mid
27. E2027: 1552.72 -> high
28. E2028: 936.26 -> mid

Let me check E2013 again: 32 * 31.26. 31.26 * 32. 31.26 * 32 = 1000.32. Yes.
E2007: 31 * 47.49 = 1472.19. Yes, <1500 so mid.
E2022: 1485.76 < 1500, so mid.
E2027: 1552.72 >= 1500, so high.
E2003: 1572.12 >= 1500, high.
E2006: 1666.80 high.
E2014: 1614.90 high.

All good.

Now format the Markdown table. The columns: ID, Hours, Rate, Pay, Band.
The header row is included? The prompt says: "Produce a Markdown table with exactly these columns, in this order: | ID | Hours | Rate | Pay | Band |"
It also says: "Output one row per input line, all 28 of them, in the original order. Output the table and nothing else — no heading, no explanation, no note about the remaining rows."
Wait: "Output the table and nothing else — no heading, no explanation, no note about the remaining rows." Does "no heading" mean no column header row? Or no overall heading? Usually "no heading" in such prompts means no title outside the table, but the table itself has a header row. Let me read carefully: "Output one row per input line, all 28 of them, in the original order. Output the table and nothing else — no heading, no explanation, no note about the remaining rows." It says "no heading", which might mean no table header row? But it also says "Produce a Markdown table with exactly these columns, in this order: | ID | Hours | Rate | Pay | Band |". That implies the header row is part of the table. Typically, "no heading" means no surrounding text, not necessarily no column headers. But to be safe, I'll include the column header row because it says "with exactly these columns, in this order: | ID | Hours | Rate | Pay | Band |". That's the table header. I'll include it. The phrase "Output the table and nothing else — no heading, no explanation, no note about the remaining rows." likely means no prose, no comments, just the table markdown. I'll include the header separator and rows.

Let me format:
| ID | Hours | Rate | Pay | Band |
|---|---|---|---|---|
Then 28 rows.

I need to make sure Pay is rounded to two decimal places. All my computed Pays are already to 2 decimal places. I'll format them with exactly two decimals, e.g., 754.00, 420.84, etc.

Let me list all rows with proper formatting:

| E2001 | 40 | 18.85 | 754.00 | mid |
| E2002 | 18 | 23.38 | 420.84 | low |
| E2003 | 33 | 47.64 | 1572.12 | high |
| E2004 | 29 | 18.25 | 529.25 | mid |
| E2005 | 23 | 41.83 | 962.09 | mid |
| E2006 | 45 | 37.04 | 1666.80 | high |
| E2007 | 31 | 47.49 | 1472.19 | mid |
| E2008 | 41 | 23.40 | 959.40 | mid |
| E2009 | 14 | 60.80 | 851.20 | mid |
| E2010 | 22 | 34.24 | 753.28 | mid |
| E2011 | 29 | 33.70 | 977.30 | mid |
| E2012 | 18 | 20.44 | 367.92 | low |
| E2013 | 32 | 31.26 | 1000.32 | mid |
| E2014 | 42 | 38.45 | 1614.90 | high |
| E2015 | 15 | 36.11 | 541.65 | mid |
| E2016 | 13 | 25.50 | 331.50 | low |
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

Wait, for Rate column, the input has rates like 23.4, 60.8, 33.7, 25.5. Should I format them as given or with two decimals? The prompt says the columns are ID, Hours, Rate, Pay, Band. It doesn't specify formatting for Hours and Rate beyond what's in the data. I'll output them exactly as in the input, but I need to be careful: the input has `23.4`, `60.8`, `33.7`, `25.5`. I'll keep them as is, or maybe add a trailing zero? The prompt says "Produce a Markdown table with exactly these columns, in this order: | ID | Hours | Rate | Pay | Band |". It doesn't specify rounding for Hours and Rate, only for Pay: "Pay is hours multiplied by rate, rounded to two decimal places." So I'll output Hours and Rate as they appear in the input. But to be safe and consistent, I could format them to two decimal places as well, but the input has varying decimal places. I'll just copy exactly: E2008 has rate 23.4, E2009 has 60.8, E2011 has 33.7, E2016 has 25.5. I'll keep them as is. However, in markdown tables, alignment might be affected, but it's fine.

Let me check the input data again:
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

I'll output Rate exactly as given. Hours exactly as given.

Now, the header row: "Produce a Markdown table with exactly these columns, in this order: | ID | Hours | Rate | Pay | Band |" - this is the header. I'll include it. Then a separator row. Then the 28 data rows.

I need to make sure the Pay values are correctly rounded to two decimal places. I'll format them as `X.XX`.

Let me re-verify all Pay calculations one more time, maybe with a different approach or at least spot-check critical ones.

1. 40 * 18.85 = 754.00 ok
2. 18 * 23.38 = 420.84 ok
3. 33 * 47.64 = 1572.12 ok
4. 29 * 18.25 = 529.25 ok
5. 23 * 41.83 = 962.09 ok
6. 45 * 37.04 = 1666.80 ok
7. 31 * 47.49 = 1472.19 ok
8. 41 * 23.4 = 959.40 ok
9. 14 * 60.8 = 851.20 ok
10. 22 * 34.24 = 753.28 ok
11. 29 * 33.7 = 977.30 ok
12. 18 * 20.44 = 367.92 ok
13. 32 * 31.26 = 1000.32 ok
14. 42 * 38.45 = 1614.90 ok
15. 15 * 36.11 = 541.65 ok
16. 13 * 25.5 = 331.50 ok
17. 36 * 22.03 = 793.08 ok
18. 39 * 31.29 = 1220.31 ok
19. 15 * 59.02 = 885.30 ok
20. 32 * 37.36 = 1195.52 ok
21. 20 * 24.76 = 495.20 ok
22. 32 * 46.43 = 1485.76 ok
23. 44 * 59.68 = 2625.92 ok
24. 32 * 33.06 = 1057.92 ok
25. 13 * 27.41 = 356.33 ok
26. 33 * 20.69 = 682.77 ok
27. 26 * 59.72 = 1552.72 ok
28. 26 * 36.01 = 936.26 ok

All good.

Bands:
Pay < 500: low
500 <= Pay < 1500: mid
Pay >= 1500: high

Check boundaries:
- 420.84 < 500 -> low
- 367.92 < 500 -> low
- 331.50 < 500 -> low
- 495.20 < 500 -> low
- 962.09: 500 <= 962.09 < 1500 -> mid
- 529.25 -> mid
- 754.00 -> mid
- 851.20 -> mid
- 753.28 -> mid
- 977.30 -> mid
- 1000.32 -> mid
- 541.65 -> mid
- 959.40 -> mid
- 885.30 -> mid
- 1195.52 -> mid
- 1485.76 -> mid (since <1500)
- 1057.92 -> mid
- 682.77 -> mid
- 936.26 -> mid
- 1472.19 -> mid
- 1220.31 -> mid
- 885.30 -> mid (already counted)
- 541.65 -> mid
- 793.08 -> mid
- 1614.90 -> high (>=1500)
- 1572.12 -> high
- 1666.80 -> high
- 2625.92 -> high
- 1552.72 -> high

All band assignments correct.

Now format the table. I'll include the header row as `| ID | Hours | Rate | Pay | Band |` and separator `|---|---|---|---|---|` then the rows.

I need to output exactly the table and nothing else. I'll make sure no extra spaces or text.

Let me write the final table in the answer. I'll put it inside <answer> tags as requested, but the content should be just the table. The prompt says: "Output one row per input line, all 28 of them, in the original order. Output the table and nothing else — no heading, no explanation, no note about the remaining rows." And "Think first if you need to, then put your complete final answer between <answer> and </answer> tags. Everything inside those tags must be the answer itself, with no commentary about how you produced it."

So I'll output the markdown table inside the tags.

One thing: should I include the header row? The prompt says "Produce a Markdown table with exactly these columns, in this order: | ID | Hours | Rate | Pay | Band |". That is the column specification. It also says "Output the table and nothing else — no heading, no explanation, no note about the remaining rows." I think "no heading" means no surrounding text, not necessarily no column header. In markdown tables, the header row is part of the table. I'll include it. If they meant no header row, they'd usually say "no header row" or "without the header". But "no heading" is ambiguous. I'll include the header row as it's standard for "a Markdown table with exactly these columns". I'll format it as:

| ID | Hours | Rate | Pay | Band |
|---|---|---|---|---|
Then the 28 rows.

I'll double-check the format. Some people omit the header, but the prompt explicitly lists the columns in that order with pipes, which strongly implies the header row. I'll include it.

Now, I'll construct the exact markdown.

Let me write out all rows carefully:

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

I'll make sure there are no trailing spaces, and the formatting is clean.

One minor thing: In the Rate column, some values have one decimal (23.4, 60.8, 33.7, 25.5). I'll keep them as in the input. The Pay column I'll format to two decimal places always.

I think this is complete. I'll output the table inside <answer> tags.

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

