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

- Score: 9.0/10 — Response correctly explains IEEE 754 representation, non-terminating binary for 0.1/0.2, shows why sum differs, notes exact dyadic rationals, gives tolerance and exact-type approaches, and cites money as wrong use case, with clear concise wording under limit.
- Tokens/s: 56.0
- Prefill tok/s: 1257.8
- TTFT: 0.27s
- Gen Time: 70.78s
- Output Tokens: 3949
- answer read from `answer_tag`

<details><summary>Reasoning (7182 chars, not scored)</summary>

We need answer user asks explain why 0.1+0.2==0.3 false, cover 5 points, under 500 words, think first then final between answer tags. Need ensure final only answer tags? They say put complete final answer between  tags. Everything inside tags must be answer itself, no commentary. We can include concise explanation. Need under 500 words. Let's craft.

Need cover:
1 IEEE 754 double precision represents number: sign, exponent, mantissa (fraction) 52 bits plus implicit leading bit, finite range, spacing. 0.1 decimal = 1/10 not finite binary fraction because denominator 10 includes factor 5; binary fractions have denominators powers of 2. So 0.1 rounded to nearest double: 0.1000000000000000055511151231257827021181583404541015625? Actually double value is 0.1000000000000000055511151231257827021181583404541015625. 0.2 similarly.
2 0.1+0.2 produces rounded sum of two approximations. Exact sum of stored values? 0.1 double + 0.2 double = 0.3000000000000000444089209850062616169452667236328125? Let's verify. In Python repr 0.1+0.2 = 0.30000000000000004. The exact sum of the two double values is 0.3000000000000000444089209850062616169452667236328125? The nearest double to that exact sum is 0.3000000000000000444089209850062616169452667236328125? Wait double 0.3 is 0.299999999999999988897769753748434595763683319091796875. The sum result is next representable double above 0.3? Let's compute: 0.1 double = 0x1.921fb5444251cp-4? 0.2 = 0x1.921fb5444251cp-3? Sum exact = 0.3000000000000000444089209850062616169452667236328125, which is representable? It is a double? Yes 0.3000000000000000444089209850062616169452667236328125 is the double for 0.30000000000000004. The literal 0.3 is rounded to 0.299999999999999988897769753748434595763683319091796875. So not equal.
Need maybe mention addition rounds to nearest representable double; result differs from literal 0.3 because 0.3 literal is a different double.
3 0.5 + 0.25 == 0.75 true because 0.5=2^-1, 0.25=2^-2, 0.75=3/4 = binary 0.11, all finite binary fractions exactly representable. Sum exact and representable.
4 Two correct approaches for comparing floating-point values, and when appropriate.
- Relative/absolute tolerance: compare abs(a-b) <= max(abs_tol, rel_tol*max(abs(a),abs(b))) or math.isclose. Appropriate for general numeric results where small relative error expected; choose tolerance based on operations and magnitudes.
- Exact comparison for values known to be exactly representable or same computation/order, e.g., integers within exact range, binary fractions, or comparing to 0? More generally use exact equality when you need bit-exact reproducibility or values are from same deterministic computation and no rounding differences expected; or use Decimal/fixed-point for exact decimal comparisons. Need two approaches: tolerance and exact when representable/same source. Could also use integer scaling for currency. But question asks two correct approaches for comparing floating-point values, and when each appropriate. Approach 1: tolerance (absolute/relative). Approach 2: exact equality only when values are guaranteed exact or compare bit patterns? Maybe use `a == b` if both are results of same operation or known exact representable; otherwise not. Or use `math.isclose` vs `==`. Need be precise.
5 One category of application where binary floating point wrong tool, and what to use instead. Financial/currency calculations requiring exact decimal cents; use decimal fixed-point (e.g., Python Decimal, Java BigDecimal) or integer cents. Could also scientific? But category: money. Use decimal arithmetic or integer representation.

Need under 500 words. Let's draft around 350-450 words.

Potential final:


Check word count: Let's count roughly. Paragraph1: In(1) IEEE2 754(3?) double4 precision5 a6 finite7 number8 is9 stored10 as11 sign12 exponent13 and14 52-bit15 fraction16 (plus17 implicit18 leading19 1)20 so21 values22 are23 spaced24 by25 powers26 of27 two28. A29 decimal30 fraction31 is32 exact33 in34 binary35 only36 if37 its38 denominator39 is40 a41 power42 of43 two44. 0.145 =46 1/1047 has48 factor49 5,50 so51 it52 repeats53 in54 binary55 and56 is57 rounded58 to59 the60 nearest61 double,62 about63 0.100...64. 0.265 is66 similarly67 rounded68.
P2: Adding69 them70 does71 not72 add73 the74 mathematical75 decimals;76 it77 adds78 the79 two80 stored81 approximations82 and83 rounds84 the85 exact86 sum87 to88 the89 nearest90 double91. The92 result93 is94 the95 double96 0.300...97 while98 the99 literal100 0.3101 is102 rounded103 to104 a105 different106 double,107 0.299...108. Therefore109 `0.1110 +111 0.2112 ==113 0.3`114 compares115 two116 different117 binary64118 values119 and120 is121 False122.
P3: `0.5123 +124 0.25125 ==126 0.75`127 is128 true129 because130 0.5131 (2^-1)132 0.25133 (2^-2)134 and135 0.75136 (3/4137 =138 binary139 0.11)140 are141 finite142 binary143 fractions144. They145 are146 exactly147 representable,148 and149 their150 sum151 is152 exactly153 representable,154 so155 no156 rounding157 changes158 the159 result160.
P4: For161 comparisons,162 use163 a164 tolerance165 when166 values167 are168 results169 of170 floating-point171 computation:172 e.g.173 `abs(a-b)174 <=175 max(abs_tol,176 rel_tol*max(abs(a),abs(b)))`177 or178 `math.isclose`179. This180 is181 appropriate182 when183 small184 relative185 or186 absolute187 error188 is189 expected190 and191 you192 care193 about194 numerical195 closeness196. Use197 exact198 equality199 only200 when201 equality202 is203 meaningful204 at205 the206 bit207 level208 or209 the210 values211 are212 guaranteed213 exact:214 integers215 within216 exact217 range,218 binary219 fractions,220 or221 values222 produced223 by224 the225 same226 deterministic227 operation/order228. Do229 not230 use231 exact232 equality233 for234 general235 decimal-derived236 computations237.
P5: Binary238 floating239 point240 is241 the242 wrong243 tool244 for245 applications246 requiring247 exact248 decimal249 precision,250 especially251 money/currency252. Use253 decimal254 fixed-point255 types256 such257 as258 Python259 `Decimal`,260 Java261 `BigDecimal`,262 or263 store264 amounts265 as266 integers267 in268 the269 smallest270 currency271 unit272 (e.g.,273 cents)274 and275 perform276 exact277 integer278 arithmetic279.
Under 500. Good.

Need ensure no commentary outside tags. Final only tags. Could include code? Fine. Need maybe mention IEEE 754 double precision representation: sign, exponent, mantissa. Good.

Double-check exact values: 0.1 double exact decimal: 0.1000000000000000055511151231257827021181583404541015625. 0.3 double exact: 0.299999999999999988897769753748434595763683319091796875. Sum result exact? The double 0.30000000000000004 exact decimal: 0.3000000000000000444089209850062616169452667236328125. Yes.

Need maybe "most programming languages" because Python, JS etc. Good.

Potential issue: "exact equality only when equality is meaningful at bit level or values guaranteed exact" maybe too broad. But okay. Could mention `a == b` for same computation. Good.

Need final under 500 words. The answer inside tags maybe word count includes tags? Probably not. Under 500. Good.

Let's produce final.
</think>

</details>

**Answer (as scored):**

In IEEE 754 double precision, a finite number is stored as a sign, exponent, and 52-bit fraction (plus an implicit leading 1), so values are spaced by powers of two. A decimal fraction is exact in binary only if its denominator is a power of two. `0.1 = 1/10` has a factor of 5, so it repeats in binary and is rounded to the nearest double, about `0.10000000000000000555`. `0.2` is similarly rounded.

Adding them does not add the mathematical decimals; it adds the two stored approximations and rounds the exact sum to the nearest double. The result is the double `0.3000000000000000444089...`, while the literal `0.3` is rounded to a different double, `0.2999999999999999888977...`. Therefore `0.1 + 0.2 == 0.3` compares two different binary64 values and is False.

`0.5 + 0.25 == 0.75` is true because `0.5` (`2^-1`), `0.25` (`2^-2`), and `0.75` (`3/4`, binary `0.11`) are finite binary fractions. They are exactly representable, and their sum is exactly representable, so no rounding changes the result.

For comparisons, use a tolerance when values are results of floating-point computation: for example, `abs(a-b) <= max(abs_tol, rel_tol*max(abs(a),abs(b)))` or `math.isclose`. This is appropriate when small relative or absolute error is expected and you care about numerical closeness. Use exact equality only when equality is meaningful at the bit level or the values are guaranteed exact: integers within exact range, binary fractions, or values produced by the same deterministic operation/order. Do not use exact equality for general decimal-derived computations.

Binary floating point is the wrong tool for applications requiring exact decimal precision, especially money/currency. Use decimal fixed-point types such as Python `Decimal`, Java `BigDecimal`, or store amounts as integers in the smallest currency unit (e.g., cents) and perform exact integer arithmetic.

