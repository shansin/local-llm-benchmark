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

- Score: 9.25/10 — Response correctly explains IEEE 754 representation, non-terminating binary for 0.1, sum producing 0.30000000000000004 vs 0.3, exact dyadic case, tolerance and decimal approaches, and money use case, within word limit and clear.
- Tokens/s: 208.1
- Prefill tok/s: 2277.8
- TTFT: 0.12s
- Gen Time: 10.65s
- Output Tokens: 2190
- answer read from `answer_tag`

<details><summary>Reasoning (4150 chars, not scored)</summary>

The user wants me to explain why `0.1 + 0.2 == 0.3` evaluates to False. Let me think through the technical details carefully.

**1. IEEE 754 double-precision representation**

IEEE 754 double precision uses 64 bits: 1 sign bit, 11 exponent bits, 52 mantissa (fraction) bits. A number is represented as:

(-1)^sign × 1.mantissa × 2^(exponent - 1023)

The mantissa has an implicit leading 1 (for normalized numbers), so effectively 53 bits of precision.

The key issue: only numbers that are exact sums of powers of 2 can be represented exactly. 0.1 in decimal = 1/10, which is not a sum of powers of 2 (since 10 = 2×5, and the factor of 5 means it's a repeating fraction in binary). So 0.1 must be rounded to the nearest representable double.

0.1 in binary is 0.00011001100110011... (repeating 0011). The nearest double to 0.1 is actually 0.1000000000000000055511151231257827021181583404541015625.

**2. What 0.1 + 0.2 produces**

0.1 rounds to nearest double ≈ 0.1000000000000000055511151231257827...
0.2 rounds to nearest double ≈ 0.2000000000000000111022302462515654...

Their sum, rounded to nearest double, gives 0.30000000000000004 (approximately 0.30000000000000004440892098500626...).

0.3 as a double is ≈ 0.2999999999999999888977697537484345957636...

So 0.1 + 0.2 = 0.30000000000000004 which is not equal to 0.3's representation. Hence False.

**3. Why 0.5 + 0.25 == 0.75 is exactly true**

0.5 = 2^-1, 0.25 = 2^-2, 0.75 = 2^-1 + 2^-2. All are exact sums of powers of 2, so they're represented exactly. Their sum is also exact and equals 0.75 exactly. No rounding error.

**4. Two approaches for comparing floats**

a) Absolute tolerance (epsilon comparison): `abs(a - b) < eps`. Good when values are near zero or when you know the scale. Simple.

b) Relative tolerance: `abs(a - b) <= rel_eps * max(abs(a), abs(b))`. Good when values span large magnitudes; scales with the size of the numbers. Often combined: `abs(a-b) <= atol + rtol*max(|a|,|b|)` (like numpy.isclose).

Also could mention comparing with a small absolute epsilon for values near zero.

**5. Application where binary float is wrong**

Financial/money calculations. Use decimal arithmetic (e.g., Python's `decimal.Decimal`, or integer cents). Money needs exact base-10 representation.

Let me write this concisely under 500 words.

Let me draft:

---

**1. IEEE 754 double precision**

A double uses 64 bits: 1 sign, 11 exponent, 52 mantissa bits (53 with the implicit leading 1). Values are stored as ±1.mantissa × 2^exponent. Only numbers expressible as exact sums of powers of 2 fit; everything else is rounded to the nearest representable value.

0.1 = 1/10 is not such a number. In binary it's 0.0001100110011... with 0011 repeating infinitely. The nearest double is 0.1000000000000000055511151231257827..., not exactly 0.1.

**2. What 0.1 + 0.2 produces**

0.1 and 0.2 each round up slightly. Adding them and rounding to the nearest double yields 0.30000000000000004. The stored value of the literal 0.3 is 0.2999999999999999888977697537484345..., a different double. Since 0.30000000000000004 ≠ 0.299999999999999988..., the equality is False.

**3. Why 0.5 + 0.25 == 0.75 is exact**

0.5 = 2⁻¹, 0.25 = 2⁻², 0.75 = 2⁻¹ + 2⁻² are all exact sums of negative powers of 2, so each is represented without rounding. Their sum is also exact and equals 0.75 bit-for-bit.

**4. Comparing floats correctly**

- Absolute tolerance: `abs(a - b) < eps`. Simple and appropriate when values are near zero or share a known small scale.
- Relative tolerance: `abs(a - b) <= rtol × max(|a|, |b|)`. Appropriate when magnitudes vary widely, since the allowed error scales with the numbers. Often combined as `abs(a-b) <= atol + rtol·max(|a|,|b|)` (e.g., numpy.isclose).

**5. Where binary float fails**

Financial/money code. Currency requires exact decimal fractions (0.10, 0.20), which binary floats can't represent. Use decimal arithmetic (Python's `decimal.Decimal`) or integer cents instead.

---

Let me count words roughly. That's around 250-280 words. Good, well under 500.

Let me refine and make sure it's accurate and complete. I think this is good. Let me finalize.
</think>

</details>

**Answer (as scored):**

**1. IEEE 754 double precision**

A double uses 64 bits: 1 sign, 11 exponent, 52 mantissa bits (53 with the implicit leading 1). Values are stored as ±1.mantissa × 2^exponent. Only numbers expressible as exact sums of powers of 2 fit; everything else is rounded to the nearest representable value.

0.1 = 1/10 is not such a number. In binary it is 0.0001100110011… with 0011 repeating infinitely. The nearest double is 0.1000000000000000055511151231257827…, not exactly 0.1.

**2. What 0.1 + 0.2 produces**

Both 0.1 and 0.2 round *up* slightly from their true values. Adding them and rounding to the nearest double yields 0.30000000000000004. The stored value of the literal 0.3 is 0.2999999999999999888977697537484345…, a *different* double. Since 0.30000000000000004 ≠ 0.299999999999999988…, the equality is False.

**3. Why 0.5 + 0.25 == 0.75 is exactly true**

0.5 = 2⁻¹, 0.25 = 2⁻², and 0.75 = 2⁻¹ + 2⁻² are all exact sums of negative powers of 2, so each is stored without rounding. Their sum is likewise exact and equals 0.75 bit-for-bit.

**4. Comparing floats correctly**

- **Absolute tolerance:** `abs(a - b) < eps`. Simple and appropriate when values are near zero or share a known small scale.
- **Relative tolerance:** `abs(a - b) <= rtol × max(|a|, |b|)`. Appropriate when magnitudes vary widely, since the allowed error scales with the numbers' size. Often combined as `abs(a-b) <= atol + rtol·max(|a|,|b|)` (e.g., `numpy.isclose`).

**5. Where binary float fails**

Financial / money code. Currency requires exact decimal fractions like 0.10 and 0.20, which binary floats cannot represent. Use decimal arithmetic (Python's `decimal.Decimal`) or integer cents instead.

