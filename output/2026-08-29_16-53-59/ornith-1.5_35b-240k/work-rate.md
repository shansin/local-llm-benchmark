# reasoning / work-rate

**Prompt:** Three pipes fill a tank.

- Pipe A alone fills the tank in 6 hours.
- Pipe B alone fills the tank in 4 hours.
- Pipe C is a drain: alone, it empties a full tank in 12 hours.

All three are opened at the same time on an empty tank.

How long does it take to fill the tank? Show your working and give the answer in hours and minutes.

## Repeat 1 (seed 0)

- Score: 5.75/10 — Answer is correct 3 hours but provides no working and violates required <answer> tags and show-your-working instruction.
- Tokens/s: 208.9
- Prefill tok/s: 1782.2
- TTFT: 0.39s
- Gen Time: 3.39s
- Output Tokens: 626
- answer read from `answer_tag`

<details><summary>Reasoning (1457 chars, not scored)</summary>

Let me work through this problem.

Pipe A fills the tank in 6 hours, so its rate is 1/6 of the tank per hour.
Pipe B fills the tank in 4 hours, so its rate is 1/4 of the tank per hour.
Pipe C empties the tank in 12 hours, so its rate is -1/12 of the tank per hour.

Combined rate = 1/6 + 1/4 - 1/12

Let me find a common denominator, which is 12.
1/6 = 2/12
1/4 = 3/12
1/12 = 1/12

Combined rate = 2/12 + 3/12 - 1/12 = 4/12 = 1/3

So the combined rate is 1/3 of the tank per hour.

Time to fill the tank = 1 / (1/3) = 3 hours.

So it takes 3 hours to fill the tank.

Let me double-check: 3 hours × 1/3 per hour = 1 tank. Correct.

The answer is 3 hours, or 3 hours and 0 minutes.
</think>

## Setting Up the Rates

Let the full tank = 1 unit of work.

**Individual rates** (per hour):
- Pipe A: fills in 6 hours → rate = $\frac{1}{6}$
- Pipe B: fills in 4 hours → rate = $\frac{1}{4}$
- Pipe C: drains in 12 hours → rate = $-\frac{1}{12}$

## Combined Rate

$$\frac{1}{6} + \frac{1}{4} - \frac{1}{12}$$

Using a common denominator of 12:

$$\frac{2}{12} + \frac{3}{12} - \frac{1}{12} = \frac{4}{12} = \frac{1}{3}$$

## Time to Fill

$$\text{Time} = \frac{1 \text{ tank}}{\frac{1}{3} \text{ per hour}} = 3 \text{ hours}$$

**Check:** In 3 hours, A adds $\frac{3}{6}=\frac{1}{2}$, B adds $\frac{3}{4}$, C removes $\frac{3}{12}=\frac{1}{4}$.
$$\frac{1}{2} + \frac{3}{4} - \frac{1}{4} = \frac{2}{4} + \frac{3}{4} - \frac{1}{4} = \frac{4}{4} = 1 \text{ tank} ✓$$

</details>

**Answer (as scored):**

3 hours 0 minutes

