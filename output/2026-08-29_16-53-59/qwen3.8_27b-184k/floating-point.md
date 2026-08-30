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

- Score: 9.0/10 — Response correctly explains IEEE 754 representation, rounding of 0.1/0.2, exact dyadic case, tolerance strategies and money use case, with minor nuance on epsilon description but overall accurate and clear within word limit.
- Tokens/s: 82.6
- Prefill tok/s: 1037.5
- TTFT: 10.24s
- Gen Time: 20.14s
- Output Tokens: 1641
- answer read from `answer_tag`

<details><summary>Reasoning (2149 chars, not scored)</summary>

The user wants a concise explanation (< 500 words) of why 0.1 + 0.2 != 0.3 in floating point, covering 5 specific points. Let me structure this carefully.

1. IEEE 754 double-precision: 64 bits = 1 sign + 11 exponent + 52 mantissa. Numbers are stored as (-1)^s × 2^e × (1 + m/2^52). 0.1 in binary is 0.0001100110011... repeating. It can't be represented exactly with 52 bits of mantissa. So 0.1 is stored as the closest representable value, which is slightly more than 0.1.

2. 0.1 + 0.2: The stored values are approximations. 0.1 ≈ 0.1000000000000000055511151231257827021181583404541015625, 0.2 ≈ 0.200000000000000011102230246251565404236316680908203125. Their sum ≈ 0.3000000000000000444089209850062616169452667236328125. But 0.3 is stored as ≈ 0.299999999999999988897769753748434595763683319091796875. So they differ.

3. 0.5 + 0.25 == 0.75: 0.5 = 2^-1, 0.25 = 2^-2, 0.75 = 2^-1 + 2^-2. All are exact powers of 2 or sums of powers of 2, so they're exactly representable.

4. Two approaches:
   - Epsilon comparison: abs(a - b) < epsilon. Good for general-purpose comparisons where you know the scale.
   - Relative comparison: abs(a - b) < epsilon * max(abs(a), abs(b)). Better when values can be very large or very small.
   - Or: use a tolerance relative to the magnitude.
   
   Actually, let me think about "two correct approaches":
   - Absolute epsilon: |a - b| < ε. Appropriate when values are known to be in a bounded range (e.g., probabilities between 0 and 1).
   - Relative epsilon: |a - b| < ε * max(|a|, |b|). Appropriate when values span many orders of magnitude.
   
   Or another pair:
   - Epsilon comparison for general floating point
   - Exact integer/rational arithmetic (e.g., using Decimal, fractions, or integer cents) when you need exactness

   I think the question asks for two approaches to *comparing* floating-point values specifically. Let me go with absolute epsilon and relative epsilon.

5. Category where binary FP is wrong: Financial/money calculations. Use decimal arithmetic (e.g., Python's `decimal` module, Java's `BigDecimal`, or integer arithmetic in cents).

Let me write this concisely under 500 words.

</details>

**Answer (as scored):**

**1. Why 0.1 isn't exact in IEEE 754 double-precision**

A double uses 64 bits: 1 sign, 11 exponent, 52 mantissa. A value is stored as (−1)^s × 2^e × (1 + m/2^52). This means only numbers whose fractional part is a sum of negative powers of 2 can be represented exactly. 0.1 in binary is 0.0001100110011… repeating forever. With only 52 mantissa bits, the stored value is the *nearest* representable double: ≈ 0.1000000000000000055511151231257827. It is slightly *more* than 0.1.

**2. What 0.1 + 0.2 actually produces**

The stored 0.1 is ≈ 0.1000000000000000055511 and the stored 0.2 is ≈ 0.2000000000000000111022. Their exact sum is ≈ 0.3000000000000000444089. But the literal 0.3 is stored as the nearest double to 0.3, which is ≈ 0.2999999999999999888978. These two doubles differ in the last few ulps, so `==` returns False. The error isn't in the addition itself; it's that the three literals round to different grid points on the 2^−52 lattice.

**3. Why 0.5 + 0.25 == 0.75 is exactly true**

0.5 = 2⁻¹, 0.25 = 2⁻², and 0.75 = 2⁻¹ + 2⁻². Each is a finite sum of negative powers of 2, so each fits exactly in the 52-bit mantissa. No rounding occurs at any step, and the equality holds bit-for-bit.

**4. Two correct comparison strategies**

- **Absolute epsilon:** `abs(a − b) < ε` (e.g., ε = 1e−9). Appropriate when your values live in a known, bounded range—probabilities, normalized coordinates, sensor readings near 1.0. You pick ε based on the problem's tolerance.

- **Relative epsilon:** `abs(a − b) < ε × max(abs(a), abs(b))`. Appropriate when magnitudes span many orders of magnitude (scientific simulation, financial time-series at different scales). It scales the tolerance to the size of the operands, so a comparison at 10⁶ doesn't demand the same absolute precision as one at 10⁻⁶.

Both are heuristics; neither is "correct" in a mathematical sense. Choose based on the error model of your computation.

**5. Where binary floating point is the wrong tool**

**Monetary and accounting calculations.** Cents, interest accrual, tax brackets, and ledger balances require exact decimal arithmetic. Binary FP will silently drift over thousands of transactions. Use a decimal type instead: Python's `decimal.Decimal`, Java's `BigDecimal`, C#'s `decimal`, or simply integer arithmetic in the smallest currency unit (e.g., store amounts as integer cents). These types represent base-10 fractions exactly and support configurable rounding modes, eliminating the class of bugs that `0.1 + 0.2` exposes.

