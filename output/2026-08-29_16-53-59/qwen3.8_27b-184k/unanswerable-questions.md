# faithfulness / unanswerable-questions

**Prompt:** Answer the seven questions below using only the passage. If the passage does not contain
the information needed to answer a question, the answer is the exact string "not stated".
Do not use outside knowledge and do not infer a figure the passage does not give.

PASSAGE
The Kestrel Point wind farm was commissioned in March 2019 with 14 turbines. Its operator,
Marrow Energy, reported an average capacity factor of 41% across the site's first full year
of operation. In 2021 the site added a 20 MW battery installation. Following a subsea cable
fault in November 2022, output from the farm was curtailed for nine weeks while repairs were
carried out. The site employs 23 people on a permanent basis, with additional seasonal
contractors during maintenance windows.

QUESTIONS
1. How many turbines were commissioned in March 2019?
2. What average capacity factor was reported for the first full year?
3. How many weeks was output curtailed after the cable fault?
4. How many people does the site employ on a permanent basis?
5. How much did the wind farm cost to build?
6. Which manufacturer supplied the turbines?
7. How many hours of energy can the battery installation store?

Return only a JSON object, with no code fence and no commentary:

{"q1": <answer>, "q2": <answer>, "q3": <answer>, "q4": <answer>, "q5": <answer>, "q6": <answer>, "q7": <answer>}

Numeric answers should be bare numbers. Any answer the passage does not support must be the
exact string "not stated".

## Repeat 1 (seed 0)

- Score: 9.5/10 — All answers are correct except q2 format uses '41%' instead of bare number 41, otherwise faithfulness and JSON are perfect.
- Tokens/s: 98.9
- Prefill tok/s: 1300.9
- TTFT: 7.72s
- Gen Time: 8.20s
- Output Tokens: 772
- answer read from `answer_tag`

<details><summary>Reasoning (2520 chars, not scored)</summary>

Let me analyze each question against the passage:

1. "How many turbines were commissioned in March 2019?" - The passage states: "The Kestrel Point wind farm was commissioned in March 2019 with 14 turbines." Answer: 14

2. "What average capacity factor was reported for the first full year?" - The passage states: "reported an average capacity factor of 41% across the site's first full year of operation." Answer: 41%

3. "How many weeks was output curtailed after the cable fault?" - The passage states: "output from the farm was curtailed for nine weeks while repairs were carried out." Answer: 9

4. "How many people does the site employ on a permanent basis?" - The passage states: "The site employs 23 people on a permanent basis" Answer: 23

5. "How much did the wind farm cost to build?" - The passage does not mention any cost. Answer: "not stated"

6. "Which manufacturer supplied the turbines?" - The passage does not mention a manufacturer. Answer: "not stated"

7. "How many hours of energy can the battery installation store?" - The passage mentions "a 20 MW battery installation" but does not state how many hours of energy it can store. 20 MW is a power rating, not an energy storage duration. Answer: "not stated"

Now for the format:
- Numeric answers should be bare numbers. So q1 = 14, q3 = 9, q4 = 23
- q2 is "41%" - this is a percentage, not a bare number. I'll include the % sign since it's part of the value. Actually, let me reconsider. The instruction says "Numeric answers should be bare numbers." 41% - is this a numeric answer? It's a percentage. I think "41%" is appropriate since the % is part of the unit. But "bare numbers" might mean just 41. Let me think... The passage says "41%" - I'll go with "41%" since removing the % would change the meaning. Actually, re-reading: "Numeric answers should be bare numbers." I think this means don't write "14 turbines" but just "14". For 41%, I'll write "41%" since the percent sign is part of the value, not a unit like "turbines" or "weeks". Hmm, but to be safe and follow "bare numbers" strictly, maybe just 41. But that loses information. I'll go with "41%" as it's the value as stated.

Wait, let me reconsider. "Bare numbers" likely means without units or extra words. So 14 not "14 turbines", 9 not "nine weeks", 23 not "23 people". For 41%, the % is part of the number representation. I'll include it as "41%".

Let me format the JSON:
{"q1": 14, "q2": "41%", "q3": 9, "q4": 23, "q5": "not stated", "q6": "not stated", "q7": "not stated"}

</details>

**Answer (as scored):**

{"q1": 14, "q2": "41%", "q3": 9, "q4": 23, "q5": "not stated", "q6": "not stated", "q7": "not stated"}

