# reasoning / twelve-balls

**Prompt:** You have 12 identical-looking balls. Exactly one is counterfeit: it differs in weight from the other 11, but you do not know whether it is heavier or lighter.

You have a balance scale that compares two groups and tells you which side is heavier, or that they are equal. You may use it exactly three times.

Describe a strategy that always identifies the counterfeit ball AND determines whether it is heavy or light, in three weighings, no matter which ball is counterfeit.

Set out the full decision tree: state what you weigh first, and then what you weigh in every possible outcome branch.

## Repeat 1 (seed 0)

- Score: 9.75/10 — Strategy is correct, covers all 24 possibilities with three weighings, mixes groups in unbalanced branch, and decision tree is explicit and clear.
- Tokens/s: 112.1
- Prefill tok/s: 956.5
- TTFT: 31.11s
- Gen Time: 47.24s
- Output Tokens: 5268
- answer read from `answer_tag`

<details><summary>Reasoning (7859 chars, not scored)</summary>

Let me work through this classic 12-ball puzzle carefully.

We have 12 balls, one is counterfeit (heavier or lighter), and we need to identify which one AND whether it's heavy or light in exactly 3 weighings.

Let me label the balls 1-12.

**Weighing 1:** Weigh {1, 2, 3, 4} vs {5, 6, 7, 8}

**Case A: They balance (1,2,3,4 = 5,6,7,8)**
This means the counterfeit is among {9, 10, 11, 12}, and balls 1-8 are all genuine.

**Weighing 2 (Case A):** Weigh {9, 10, 11} vs {1, 2, 3} (three known genuine balls)

- **Case A1: They balance** → counterfeit is ball 12.
  **Weighing 3:** Weigh {12} vs {1} (known genuine).
  - If 12 is heavier → 12 is heavy
  - If 12 is lighter → 12 is light

- **Case A2: {9,10,11} is heavier than {1,2,3}** → one of 9,10,11 is heavy.
  **Weighing 3:** Weigh {9} vs {10}.
  - If 9 > 10 → 9 is heavy
  - If 9 < 10 → 10 is heavy
  - If 9 = 10 → 11 is heavy

- **Case A3: {9,10,11} is lighter than {1,2,3}** → one of 9,10,11 is light.
  **Weighing 3:** Weigh {9} vs {10}.
  - If 9 < 10 → 9 is light
  - If 9 > 10 → 10 is light
  - If 9 = 10 → 11 is light

**Case B: {1,2,3,4} is heavier than {5,6,7,8}**
This means either one of {1,2,3,4} is heavy, or one of {5,6,7,8} is light. Balls 9-12 are genuine.

**Weighing 2 (Case B):** Weigh {1, 2, 5} vs {3, 6, 9} (9 is known genuine)

Let me think about what each outcome tells us:
- If {1,2,5} > {3,6,9}: 
  - 1 is heavy (was on heavy side, still on heavy side)
  - 2 is heavy (was on heavy side, still on heavy side)
  - 6 is light (was on light side, now on heavy side - if 6 is light, it would make the right side lighter, so left appears heavier)
  
  Wait, let me reconsider. In weighing 1, {1,2,3,4} > {5,6,7,8}. So the suspects are: 1H, 2H, 3H, 4H, 5L, 6L, 7L, 8L.

  In weighing 2: {1, 2, 5} vs {3, 6, 9}
  
  - If left > right: 
    - 1H: 1 is on left, makes left heavier ✓
    - 2H: 2 is on left, makes left heavier ✓
    - 6L: 6 is on right, makes right lighter (left appears heavier) ✓
    - 3H: 3 is on right, would make right heavier → left < right ✗
    - 4H: 4 is not on scale → balance ✗
    - 5L: 5 is on left, makes left lighter → left < right ✗
    - 7L: 7 is not on scale → balance ✗
    - 8L: 8 is not on scale → balance ✗
    
    So if left > right: {1H, 2H, 6L}

  - If left < right:
    - 3H: 3 is on right, makes right heavier ✓
    - 5L: 5 is on left, makes left lighter ✓
    - 1H: 1 is on left, makes left heavier → left > right ✗
    - 2H: 2 is on left, makes left heavier → left > right ✗
    - 6L: 6 is on right, makes right lighter → left > right ✗
    - 4H: not on scale → balance ✗
    - 7L: not on scale → balance ✗
    - 8L: not on scale → balance ✗
    
    So if left < right: {3H, 5L}

  - If left = right:
    - 4H: not on scale ✓
    - 7L: not on scale ✓
    - 8L: not on scale ✓
    - 1H: on left, would make left heavier ✗
    - 2H: on left, would make left heavier ✗
    - 3H: on right, would make right heavier ✗
    - 5L: on left, would make left lighter ✗
    - 6L: on right, would make right lighter ✗
    
    So if left = right: {4H, 7L, 8L}

  **Weighing 3 for B-left>right ({1H, 2H, 6L}):** Weigh {1} vs {2}.
  - If 1 > 2 → 1 is heavy
  - If 1 < 2 → 2 is heavy
  - If 1 = 2 → 6 is light

  **Weighing 3 for B-left<right ({3H, 5L}):** Weigh {3} vs {9} (genuine).
  - If 3 > 9 → 3 is heavy
  - If 3 = 9 → 5 is light
  (3 < 9 is impossible since 3 can only be heavy)

  **Weighing 3 for B-left=right ({4H, 7L, 8L}):** Weigh {7} vs {8}.
  - If 7 < 8 → 7 is light
  - If 7 > 8 → 8 is light
  - If 7 = 8 → 4 is heavy

**Case C: {1,2,3,4} is lighter than {5,6,7,8}**
This is symmetric to Case B. Either one of {1,2,3,4} is light, or one of {5,6,7,8} is heavy. Balls 9-12 are genuine.

**Weighing 2 (Case C):** Weigh {1, 2, 5} vs {3, 6, 9} (9 is known genuine)

Suspects: 1L, 2L, 3L, 4L, 5H, 6H, 7H, 8H.

- If left > right:
  - 5H: 5 is on left, makes left heavier ✓
  - 6H: 6 is on right, makes right heavier → left < right ✗
  - 3L: 3 is on right, makes right lighter → left > right ✓
  - 1L: 1 is on left, makes left lighter → left < right ✗
  - 2L: 2 is on left, makes left lighter → left < right ✗
  - 4L: not on scale → balance ✗
  - 7H: not on scale → balance ✗
  - 8H: not on scale → balance ✗
  
  So if left > right: {5H, 3L}

- If left < right:
  - 1L: 1 is on left, makes left lighter ✓
  - 2L: 2 is on left, makes left lighter ✓
  - 6H: 6 is on right, makes right heavier ✓
  - 5H: 5 is on left, makes left heavier → left > right ✗
  - 3L: 3 is on right, makes right lighter → left > right ✗
  - 4L: not on scale → balance ✗
  - 7H: not on scale → balance ✗
  - 8H: not on scale → balance ✗
  
  So if left < right: {1L, 2L, 6H}

- If left = right:
  - 4L: not on scale ✓
  - 7H: not on scale ✓
  - 8H: not on scale ✓
  - 1L: on left, makes left lighter ✗
  - 2L: on left, makes left lighter ✗
  - 3L: on right, makes right lighter ✗
  - 5H: on left, makes left heavier ✗
  - 6H: on right, makes right heavier ✗
  
  So if left = right: {4L, 7H, 8H}

  **Weighing 3 for C-left>right ({5H, 3L}):** Weigh {5} vs {9} (genuine).
  - If 5 > 9 → 5 is heavy
  - If 5 = 9 → 3 is light
  (5 < 9 is impossible)

  **Weighing 3 for C-left<right ({1L, 2L, 6H}):** Weigh {1} vs {2}.
  - If 1 < 2 → 1 is light
  - If 1 > 2 → 2 is light
  - If 1 = 2 → 6 is heavy

  **Weighing 3 for C-left=right ({4L, 7H, 8H}):** Weigh {7} vs {8}.
  - If 7 > 8 → 7 is heavy
  - If 7 < 8 → 8 is heavy
  - If 7 = 8 → 4 is light

Let me verify this is complete and correct. Let me double-check Case B more carefully.

Case B: {1,2,3,4} > {5,6,7,8}. Suspects: 1H, 2H, 3H, 4H, 5L, 6L, 7L, 8L.

Weighing 2: {1, 2, 5} vs {3, 6, 9}

If 1 is heavy: left has 1 (heavy) → left heavier. Right has 3 (normal), 6 (normal), 9 (normal). So left > right. ✓
If 2 is heavy: left has 2 (heavy) → left heavier. So left > right. ✓
If 3 is heavy: right has 3 (heavy) → right heavier. So left < right. ✓
If 4 is heavy: 4 not on scale. Both sides normal. So left = right. ✓
If 5 is light: left has 5 (light) → left lighter. So left < right. ✓
If 6 is light: right has 6 (light) → right lighter. So left > right. ✓
If 7 is light: 7 not on scale. Both sides normal. So left = right. ✓
If 8 is light: 8 not on scale. Both sides normal. So left = right. ✓

Great, this checks out.

Case C: {1,2,3,4} < {5,6,7,8}. Suspects: 1L, 2L, 3L, 4L, 5H, 6H, 7H, 8H.

Weighing 2: {1, 2, 5} vs {3, 6, 9}

If 1 is light: left has 1 (light) → left lighter. So left < right. ✓
If 2 is light: left has 2 (light) → left lighter. So left < right. ✓
If 3 is light: right has 3 (light) → right lighter. So left > right. ✓
If 4 is light: 4 not on scale. Both sides normal. So left = right. ✓
If 5 is heavy: left has 5 (heavy) → left heavier. So left > right. ✓
If 6 is heavy: right has 6 (heavy) → right heavier. So left < right. ✓
If 7 is heavy: 7 not on scale. Both sides normal. So left = right. ✓
If 8 is heavy: 8 not on scale. Both sides normal. So left = right. ✓

Great, this checks out too.

Now let me verify the third weighings:

Case B, left > right: {1H, 2H, 6L}. Weigh 1 vs 2.
- If 1H: 1 > 2 (since 2 is normal). ✓
- If 2H: 1 < 2 (since 1 is normal). ✓
- If 6L: 1 = 2 (both normal). ✓

Case B, left < right: {3H, 5L}. Weigh 3 vs 9 (genuine).
- If 3H: 3 > 9. ✓
- If 5L: 3 = 9 (both normal). ✓

Case B, left = right: {4H, 7L, 8L}. Weigh 7 vs 8.
- If 4H: 7 = 8 (both normal). ✓
- If 7L: 7 < 8 (8 is normal). ✓
- If 8L: 7 > 8 (7 is normal). ✓

Case C, left > right: {5H, 3L}. Weigh 5 vs 9 (genuine).
- If 5H: 5 > 9. ✓
- If 3L: 5 = 9 (both normal). ✓

Case C, left < right: {1L, 2L, 6H}. Weigh 1 vs 2.
- If 1L: 1 < 2 (2 is normal). ✓
- If 2L: 1 > 2 (1 is normal). ✓
- If 6H: 1 = 2 (both normal). ✓

Case C, left = right: {4L, 7H, 8H}. Weigh 7 vs 8.
- If 4L: 7 = 8 (both normal). ✓
- If 7H: 7 > 8 (8 is normal). ✓
- If 8H: 7 < 8 (7 is normal). ✓

Everything checks out. Let me write up the complete answer.

</details>

**Answer (as scored):**

# The 12-Ball Problem – Complete Three-Weighing Strategy

Label the balls **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12**.

---

## WEIGHING 1

**Weigh {1, 2, 3, 4} against {5, 6, 7, 8}.**

Three outcomes are possible:

---

## CASE A – The scale BALANCES (1,2,3,4 = 5,6,7,8)

Balls 1–8 are all genuine. The counterfeit is among **{9, 10, 11, 12}**, and you still don't know if it's heavy or light.

### Weighing 2 (Case A)

**Weigh {9, 10, 11} against {1, 2, 3}** (three known-genuine balls).

#### A1 – Balance (9,10,11 = 1,2,3)
The counterfeit is **ball 12**.

**Weighing 3:** Weigh **12 vs 1** (genuine).
- 12 heavier → **12 is the counterfeit, HEAVY.**
- 12 lighter → **12 is the counterfeit, LIGHT.**

#### A2 – Left side heavier (9,10,11 > 1,2,3)
One of 9, 10, 11 is **heavy**.

**Weighing 3:** Weigh **9 vs 10**.
- 9 > 10 → **9 is the counterfeit, HEAVY.**
- 9 < 10 → **10 is the counterfeit, HEAVY.**
- 9 = 10 → **11 is the counterfeit, HEAVY.**

#### A3 – Left side lighter (9,10,11 < 1,2,3)
One of 9, 10, 11 is **light**.

**Weighing 3:** Weigh **9 vs 10**.
- 9 < 10 → **9 is the counterfeit, LIGHT.**
- 9 > 10 → **10 is the counterfeit, LIGHT.**
- 9 = 10 → **11 is the counterfeit, LIGHT.**

---

## CASE B – Left side HEAVIER (1,2,3,4 > 5,6,7,8)

Either one of {1,2,3,4} is **heavy**, or one of {5,6,7,8} is **light**. Balls 9–12 are genuine.

### Weighing 2 (Case B)

**Weigh {1, 2, 5} against {3, 6, 9}** (ball 9 is known genuine).

#### B1 – Left side heavier (1,2,5 > 3,6,9)
The culprit is one of: **1 heavy, 2 heavy, or 6 light.**

**Weighing 3:** Weigh **1 vs 2**.
- 1 > 2 → **1 is the counterfeit, HEAVY.**
- 1 < 2 → **2 is the counterfeit, HEAVY.**
- 1 = 2 → **6 is the counterfeit, LIGHT.**

#### B2 – Left side lighter (1,2,5 < 3,6,9)
The culprit is one of: **3 heavy or 5 light.**

**Weighing 3:** Weigh **3 vs 9** (genuine).
- 3 > 9 → **3 is the counterfeit, HEAVY.**
- 3 = 9 → **5 is the counterfeit, LIGHT.**

#### B3 – Balance (1,2,5 = 3,6,9)
The culprit is one of: **4 heavy, 7 light, or 8 light.**

**Weighing 3:** Weigh **7 vs 8**.
- 7 < 8 → **7 is the counterfeit, LIGHT.**
- 7 > 8 → **8 is the counterfeit, LIGHT.**
- 7 = 8 → **4 is the counterfeit, HEAVY.**

---

## CASE C – Left side LIGHTER (1,2,3,4 < 5,6,7,8)

Either one of {1,2,3,4} is **light**, or one of {5,6,7,8} is **heavy**. Balls 9–12 are genuine.

### Weighing 2 (Case C)

**Weigh {1, 2, 5} against {3, 6, 9}** (ball 9 is known genuine).

#### C1 – Left side heavier (1,2,5 > 3,6,9)
The culprit is one of: **5 heavy or 3 light.**

**Weighing 3:** Weigh **5 vs 9** (genuine).
- 5 > 9 → **5 is the counterfeit, HEAVY.**
- 5 = 9 → **3 is the counterfeit, LIGHT.**

#### C2 – Left side lighter (1,2,5 < 3,6,9)
The culprit is one of: **1 light, 2 light, or 6 heavy.**

**Weighing 3:** Weigh **1 vs 2**.
- 1 < 2 → **1 is the counterfeit, LIGHT.**
- 1 > 2 → **2 is the counterfeit, LIGHT.**
- 1 = 2 → **6 is the counterfeit, HEAVY.**

#### C3 – Balance (1,2,5 = 3,6,9)
The culprit is one of: **4 light, 7 heavy, or 8 heavy.**

**Weighing 3:** Weigh **7 vs 8**.
- 7 > 8 → **7 is the counterfeit, HEAVY.**
- 7 < 8 → **8 is the counterfeit, HEAVY.**
- 7 = 8 → **4 is the counterfeit, LIGHT.**

---

## Summary Table

| W1 result | W2 result | W3 result | Counterfeit |
|---|---|---|---|
| = | = | 12 > 1 | 12 Heavy |
| = | = | 12 < 1 | 12 Light |
| = | > | 9 > 10 | 9 Heavy |
| = | > | 9 < 10 | 10 Heavy |
| = | > | 9 = 10 | 11 Heavy |
| = | < | 9 < 10 | 9 Light |
| = | < | 9 > 10 | 10 Light |
| = | < | 9 = 10 | 11 Light |
| > | > | 1 > 2 | 1 Heavy |
| > | > | 1 < 2 | 2 Heavy |
| > | > | 1 = 2 | 6 Light |
| > | < | 3 > 9 | 3 Heavy |
| > | < | 3 = 9 | 5 Light |
| > | = | 7 < 8 | 7 Light |
| > | = | 7 > 8 | 8 Light |
| > | = | 7 = 8 | 4 Heavy |
| < | > | 5 > 9 | 5 Heavy |
| < | > | 5 = 9 | 3 Light |
| < | < | 1 < 2 | 1 Light |
| < | < | 1 > 2 | 2 Light |
| < | < | 1 = 2 | 6 Heavy |
| < | = | 7 > 8 | 7 Heavy |
| < | = | 7 < 8 | 8 Heavy |
| < | = | 7 = 8 | 4 Light |

Every one of the 24 possibilities (12 balls × heavy/light) is resolved in exactly three weighings.

