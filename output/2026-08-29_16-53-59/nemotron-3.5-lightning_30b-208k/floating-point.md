# knowledge / floating-point

**Prompt:** Explain why `0.1 + 0.2 == 0.3` evaluates to False in most programming languages.

Cover:
1. How IEEE 754 double-precision represents a number, and why 0.1 has no exact representation
2. What 0.1 + 0.2 actually produces, and why it differs from 0.3
3. Why `0.5 + 0.25 == 0.75` IS exactly true
4. Two correct approaches for comparing floating-point values, and when each is appropriate
5. One category of application where binary floating point is the wrong tool, and what to use instead

Keep your answer under 500 words.

## Repeat 1 (seed 0)

- Score: 9.0/10 — Response correctly explains IEEE 754 representation, non-terminating binary for 0.1/0.2, exact dyadic rationals, tolerance vs decimal approaches, and money use case, with minor wording imprecision on exponent bias and sum value.
- Tokens/s: 269.4
- Prefill tok/s: 2203.7
- TTFT: 3.43s
- Gen Time: 4.88s
- Output Tokens: 1284
- answer read from `thinking_field`

<details><summary>Reasoning (3242 chars, not scored)</summary>

Here, the user wants a concise explanation of floating-point arithmetic issues, specifically why 0.1 + 0.2 != 0.3, within 500 words. They've outlined 5 points to cover. I need to produce a complete answer inside <answer></answer> tags, with no commentary. I'll structure it to cover all 5 points.

First, recall IEEE 754 double-precision: 64 bits, 1 sign, 11 exponent, 52 mantissa (plus implicit leading bit). Numbers are represented as sign * mantissa * 2^exponent. 0.1 in binary is a repeating fraction: 0.0001100110011... (period 1100). So it cannot be exactly represented; the closest double is slightly above or below. Similarly 0.2. When added, the sum may not equal the representation of 0.3 due to rounding. The exact sum of the two nearest doubles to 0.1 and 0.2 yields a value slightly different from the nearest double to 0.3. That's the gist.

Point 3: 0.5 and 0.25 are powers of two (2^-1 and 2^-2), so they have exact binary representations. Their sum 0.75 is also a sum of powers of two, exact.

Point 4: Two approaches: 1) Use a tolerance/epsilon comparison, e.g., abs(a-b) < epsilon, where epsilon depends on magnitude. 2) Round to a certain number of decimal places or use decimal floating point. Also, for financial, use decimal types. When each is appropriate: epsilon for scientific/engineering where precision tolerance is fine; decimal for finance.

Point 5: Application where binary floating point is wrong: financial calculations requiring exact decimal arithmetic, e.g., currency conversion, accounting. Use decimal floating point or integer arithmetic (cents).

Now, compose the answer under 500 words. I'll aim for ~350 words.

Let's draft:

"IEEE 754 double-precision encodes a floating-point number as sign × (1 + fraction) × 2^(exponent-1023). The 52-bit fraction cannot exactly represent most decimal fractions. 0.1 in binary is 0.0001100110011... recurring, so the double stored for 0.1 is the nearest approximation, slightly above the true value. Same for 0.2. When added, the exact sum of these two approximations differs from the double nearest to 0.3, typically by about 2×10⁻¹⁶ relative error. Hence 0.1+0.2==0.3 is false.

0.5 and 0.25 are 2⁻¹ and 2⁻², which have exact binary representations (a single 1 bit in the fraction). Their sum 0.75 is also exactly representable, so the equality holds.

For comparison, two approaches: (1) Epsilon comparison: |a−b| < ε, where ε scales with magnitude (e.g., 1e-9 for values near 1). Suitable for scientific data where exact decimal equality is unnecessary. (2) Round both to a fixed decimal precision or use a decimal floating-point type (e.g., Python's Decimal). Essential for financial and counting applications where rounding errors accumulate.

Binary floating point is the wrong tool for applications requiring exact decimal arithmetic, such as currency calculations, tax computations, or any domain where values are entered and stored in decimal. In those cases, use a decimal type or store integers representing the smallest unit (e.g., cents) and convert only for display."

Check word count: Let's count roughly. I'll write it out fully and then count.

But the user wants under 500 words. I'll ensure it's concise.

I'll produce the final answer now.

</details>

**Answer (as scored):**

<IEEE 754 double‑precision stores a number as sign × (1 + fraction) × 2^(exponent‑1023), with 52 bits for the fraction. Most decimal fractions, such as 0.1, have infinite binary expansions (0.1 = 0.0001100110011…₂). The 52‑bit mantissa can only approximate them, rounding to the nearest representable value—slightly above the true 0.1. The same occurs for 0.2. When the two approximations are added, the exact sum differs from the double nearest to 0.3, typically by about 2 × 10⁻¹⁶ relative error. Hence `0.1 + 0.2 == 0.3` evaluates to false.

0.5 and 0.25 are 2⁻¹ and 2⁻², which have exact binary representations (a single 1 bit in the fraction). Their sum 0.75 is also exactly representable, so `0.5 + 0.25 == 0.75` is true.

Two reliable ways to compare floating‑point values:
1. **Epsilon (tolerance) comparison**: `abs(a − b) < ε`, where ε scales with magnitude (e.g., 1e‑9 for values near 1). Appropriate for scientific or engineering data where exact decimal equality is unnecessary.
2. **Rounding or decimal types**: Round both operands to a fixed number of decimal places, or use a decimal floating‑point type (e.g., Python’s `Decimal`). Essential for financial, accounting, or any domain where values are inherently decimal.

Binary floating point is the wrong tool for applications requiring exact decimal arithmetic, such as currency conversion, tax calculations, or any system where values are entered and stored in decimal. In those contexts, use a decimal type or store integers representing the smallest unit (e.g., cents) and convert only for display.

