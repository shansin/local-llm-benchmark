# reasoning / twelve-balls

**Prompt:** You have 12 identical-looking balls. Exactly one is counterfeit: it differs in weight from the other 11, but you do not know whether it is heavier or lighter.

You have a balance scale that compares two groups and tells you which side is heavier, or that they are equal. You may use it exactly three times.

Describe a strategy that always identifies the counterfeit ball AND determines whether it is heavy or light, in three weighings, no matter which ball is counterfeit.

Set out the full decision tree: state what you weigh first, and then what you weigh in every possible outcome branch.

## Repeat 1 (seed 0)

- Score: 9.0/10 — Strategy is correct and covers all branches with three weighings; minor issue is Branch 2C/3B list impossible outcomes but still identifies counterfeit, overall essentially complete and clear.
- Tokens/s: 55.2
- Prefill tok/s: 1201.0
- TTFT: 0.26s
- Gen Time: 156.83s
- Output Tokens: 8642
- answer read from `answer_tag`

<details><summary>Reasoning (17171 chars, not scored)</summary>

We need answer user: classic 12 balls 3 weighings identify counterfeit and heavy/light. Need full decision tree. Need think carefully. Need final answer between  tags only? User says complete final answer between tags. We can include reasoning? They say think first if need, then put complete final answer between tags. Everything inside tags must be answer itself, no commentary. So final should be . Need ensure strategy correct.

We need devise decision tree. Classic solution:
First weigh 1,2,3,4 vs 5,6,7,8.
Case equal: counterfeit among 9,10,11,12. Second weigh 9,10,11 vs 1,2,3 (known good). If equal: 12 counterfeit; third weigh 12 vs 1 to determine heavy/light. If left heavy: one of 9,10,11 heavy; third weigh 9 vs 10: if equal 11 heavy else heavier is heavy. If left light: one of 9,10,11 light; third weigh 9 vs 10: if equal 11 light else lighter is light.

Case first left heavy (1-4 heavier than 5-8). Then counterfeit among 1-8, and either one of 1-4 heavy or one of 5-8 light. 9-12 good. Need second weighing to split possibilities into 3 outcomes with 3 candidates each? Classic: weigh 1,2,5 vs 3,6,9 (or 1,2,5 vs 3,6,10). Let's analyze.
Known: if first L > R, suspects: H1,H2,H3,H4,L5,L6,L7,L8 (8 possibilities). Need second weighing with 3 outcomes each max 3 possibilities. Choose weigh: 1,2,5 vs 3,6,9 (known good). Let's map:
- If balance: suspects not on scale? 4H,7L,8L? Wait scale includes 1,2,5,3,6. If equal, counterfeit among 4H,7L,8L. Third: weigh 7 vs 8. If equal -> 4 heavy. If 7 lighter than 8 -> 7 light? But if 7 light, 7 vs 8 (8 good) left lighter. If 8 light, right lighter. So yes.
- If left side heavier: possibilities? Left: 1H,2H,6L? Let's compute. Weigh L: 1,2,5; R:3,6,9. Under suspects:
1H: L heavy (1 on left) -> left heavy.
2H: left heavy.
5L: 5 on left, light -> left lighter (right heavy).
3H: 3 on right heavy -> right heavy.
6L: 6 on right light -> left heavy.
7L,8L,4H not on scale -> equal.
So left heavy corresponds {1H,2H,6L}. Third: weigh 1 vs 2. If equal -> 6 light. If left heavy -> 1 heavy. If right heavy -> 2 heavy. Good.
- If right side heavier: possibilities {3H,5L}? Wait also? Let's recalc: right heavy if 3H (right heavy) or 5L (left light). What about? 6L gives left heavy. 1H/2H left heavy. So only two possibilities. Third: weigh 5 vs 9 (known good). If equal -> 3 heavy. If 5 lighter -> 5 light. If 5 heavier impossible? But if 3 heavy, 5 good equal. If 5 light, left lighter. So okay. But need third weighing can distinguish 3H and 5L. Weigh 5 vs 9. If equal 3H; if 5 light 5L. If 5 heavy impossible (but if occurs? no). Could also weigh 3 vs 9: if equal 5L; if 3 heavy 3H. Either. Need full decision tree.

Case first right heavy (1-4 lighter than 5-8). Symmetric. Suspects: L1,L2,L3,L4,H5,H6,H7,H8. Need second weighing. Could mirror: weigh 1,2,5 vs 3,6,9? Let's map for first R heavy (left lighter). Suspects: 1L,2L,3L,4L,5H,6H,7H,8H.
Weigh 1,2,5 vs 3,6,9.
- 1L: left light -> right heavy.
- 2L: right heavy.
- 5H: left heavy.
- 3L: right light -> left heavy.
- 6H: right heavy.
- 4L,7H,8H not on scale -> equal.
So outcomes:
Left heavy: {5H,3L}. Third weigh 5 vs 9: if 5 heavy -> 5H; if equal -> 3L. (If 5 light impossible)
Right heavy: {1L,2L,6H}. Third weigh 1 vs 2: if equal -> 6H; if left lighter -> 1L; if right lighter -> 2L.
Equal: {4L,7H,8H}. Third weigh 7 vs 8: if equal -> 4L; if left heavy -> 7H; if right heavy -> 8H.
This is symmetric but not exact mirror? It works.

Need present full decision tree. Could use labels. Need ensure exactly three weighings. Need maybe mention known good balls. Need be clear.

Alternative classic: first 1-4 vs 5-8. If unequal, second 1,2,5 vs 3,6,9. We need full branches. Let's verify all possibilities.

Let's formalize.
Balls numbered 1-12.
Weighing 1: {1,2,3,4} vs {5,6,7,8}.
Outcomes:
A: equal.
B: left pan heavier.
C: right pan heavier (left lighter).

Case A:
Weighing 2: {9,10,11} vs {1,2,3} (known good).
A1 equal: Weighing 3: {12} vs {1}. If 12 heavier -> 12 heavy; if lighter -> 12 light. (Equal impossible because exactly one counterfeit; but if equal? no. Could state if equal impossible under assumptions.)
A2 left heavier: one of 9,10,11 heavy. Weighing 3: {9} vs {10}. If 9 heavier -> 9 heavy; if 10 heavier -> 10 heavy; if equal -> 11 heavy.
A3 left lighter: one of 9,10,11 light. Weighing 3: {9} vs {10}. If 9 lighter -> 9 light; if 10 lighter -> 10 light; if equal -> 11 light.

Case B (first left heavier): suspects 1H,2H,3H,4H,5L,6L,7L,8L.
Weighing 2: {1,2,5} vs {3,6,9}. (9 known good)
B1 equal: suspects 4H,7L,8L. Weighing 3: {7} vs {8}. If equal -> 4H. If 7 lighter -> 7L. If 8 lighter -> 8L.
B2 left heavier: suspects 1H,2H,6L. Weighing 3: {1} vs {2}. If equal -> 6L. If 1 heavier -> 1H. If 2 heavier -> 2H.
B3 right heavier: suspects 3H,5L. Weighing 3: {5} vs {9}. If equal -> 3H. If 5 lighter -> 5L. (If 5 heavier impossible; but if it happens? no. Could state if 5 heavier, contradiction; but under assumptions won't.)

Case C (first right heavier): suspects 1L,2L,3L,4L,5H,6H,7H,8H.
Weighing 2: {1,2,5} vs {3,6,9}. (same)
C1 equal: suspects 4L,7H,8H. Weighing 3: {7} vs {8}. If equal -> 4L. If 7 heavier -> 7H. If 8 heavier -> 8H.
C2 left heavier: suspects 5H,3L. Weighing 3: {5} vs {9}. If 5 heavier -> 5H. If equal -> 3L. (If 5 lighter impossible.)
C3 right heavier: suspects 1L,2L,6H. Weighing 3: {1} vs {2}. If equal -> 6H. If 1 lighter -> 1L. If 2 lighter -> 2L.

Need check B3: right heavier means right pan heavier. Weighing 2 left {1,2,5}, right {3,6,9}. Suspects 3H (right heavy) and 5L (left light). Third {5} vs {9}. If 5 light, left lighter (right heavier). If 3H, equal. Good.

C2: left heavier means left pan heavier. Suspects 5H (left heavy) and 3L (right light). Third {5} vs {9}. If 5 heavy, left heavier; if 3L, equal. Good.

Need maybe mention in B2 third {1} vs {2}: if equal -> 6L; if 1 heavier -> 1H; if 2 heavier -> 2H. Good.

C3 third {1} vs {2}: if equal -> 6H; if 1 lighter -> 1L; if 2 lighter -> 2L. Good.

Need ensure no ambiguity in "left" and "right". We can define left pan and right pan. Use "left side heavier" etc.

Need maybe include a table? User asks set out full decision tree. We can present as nested bullets. Need final answer only inside tags. Could include concise but complete. Need no commentary outside tags. The final answer can be long. Need ensure tags exactly.

Let's consider if there is any hidden issue: In case A, second weighing {9,10,11} vs {1,2,3}. If left heavier, one of 9,10,11 heavy. Third {9} vs {10}. If equal, 11 heavy. Good. If left lighter, one light. Third {9} vs {10}. If equal, 11 light. Good. If equal, 12. Third {12} vs {1}. If 12 heavier -> heavy; if lighter -> light. Good.

In unequal first case, we use ball 9 as known good. Are 9-12 known good? Yes because first weighing compared 1-8 and equal? Wait if first unequal, 9-12 are not weighed, but since exactly one counterfeit among 12 and first weighing unequal, counterfeit must be among weighed balls 1-8. Thus 9-12 are genuine. Good.

Need maybe mention "known good" in branches.

Could there be a more elegant decision tree with first 1-4 vs 5-8, second 1,2,5 vs 3,6,9. This is standard. Need ensure in first unequal, second weighing includes 5 (from lighter side) and 3 (from heavier side) and 6 (from lighter side) etc. Let's double-check mapping with algebra.

First L > R. Let weight normal = 0, counterfeit heavy = +d, light = -d. Weighing 1 sum L - R = sum(1-4) - sum(5-8) >0. If i in 1-4 heavy: +d >0. If i in 5-8 light: -d on right makes L-R = +d >0. Good.
Second L2-R2 = (1+2+5) - (3+6+9). 9=0.
For 1H: +d. 2H:+d. 5L: -d on left => -d. 3H: -d on right => -d? Wait if 3 heavy on right, right heavier, so L2-R2 = -d. Yes right heavy. 6L: -d on right, right lighter, L2-R2 = +d. 4H,7L,8L: 0. So left heavy {1H,2H,6L}; right heavy {3H,5L}; equal {4H,7L,8L}. Good.

First L < R. Suspects 1-4 light, 5-8 heavy.
Second same:
1L: -d left => L2-R2=-d (right heavy). 2L=-d. 5H:+d left => left heavy. 3L: -d right => L2-R2=+d (left heavy). 6H:+d right => L2-R2=-d (right heavy). 4L,7H,8H=0. So left heavy {5H,3L}; right heavy {1L,2L,6H}; equal {4L,7H,8H}. Good.

Third weighings:
B1 equal {4H,7L,8L}. Weigh 7 vs 8. If 7L: 7 lighter; if 8L: 8 lighter; if 4H: equal. Good.
C1 equal {4L,7H,8H}. Weigh 7 vs 8. If 7H: 7 heavier; if 8H: 8 heavier; if 4L: equal. Good.

B2 left heavy {1H,2H,6L}. Weigh 1 vs 2. If 1H: 1 heavier; 2H: 2 heavier; 6L: equal. Good.
C3 right heavy {1L,2L,6H}. Weigh 1 vs 2. If 1L: 1 lighter; 2L: 2 lighter; 6H: equal. Good.

B3 right heavy {3H,5L}. Weigh 5 vs 9. If 5L: 5 lighter; 3H: equal. Good.
C2 left heavy {5H,3L}. Weigh 5 vs 9. If 5H: 5 heavier; 3L: equal. Good.

All good.

Need maybe include "If an impossible result occurs, the assumptions are violated" but not necessary. Could include for completeness? The problem says no matter which ball counterfeit. We can state outcomes that can occur. For two-candidate branches, third weighing has one impossible outcome. We can mention it won't occur. But full decision tree should specify what to do in every possible outcome branch. For impossible branch, maybe not needed, but to be safe, state "If the other result occurs, it is impossible under the problem assumptions." But user asks every possible outcome branch. We can include all three scale outcomes for each third weighing, with impossible noted. But maybe too verbose. Need full decision tree. Let's include all outcomes for each weighing, marking impossible if relevant. But ensure not confusing.

Could present as:
Number balls 1-12.
Weighing 1: 1,2,3,4 vs 5,6,7,8.
Then branches.

Need maybe use "L" and "R" for left/right. Define:
- "L > R" means left pan heavier.
- "L = R" equal.
- "L < R" right pan heavier.

Then decision tree.

Let's craft final answer elegantly.

Potential issue: In case B3, third weighing 5 vs 9: if 5 is lighter, identify 5 light; if equal, identify 3 heavy. What if 5 heavier? impossible. But if we want full decision tree for all scale outcomes, include: if 5 heavier, contradiction (cannot happen). But maybe user expects every possible outcome branch, not impossible. Could include "If 5 is heavier, this cannot happen; it would indicate an error in the premises." But final answer should be strategy; we can omit impossible. However "full decision tree" might require specifying for each of three outcomes at each weighing. For branches with only two candidates, one outcome impossible. We can state it.

Let's consider if there is a way to make third weighing in two-candidate branch have all three outcomes meaningful? Not necessary. But to be thorough, include impossible.

Need ensure final answer inside 

Need check in case 3b: Weighing 2 left pan heavier: possibilities 5H,3L. Weighing 3 5 vs 9. If 5 heavier -> 5 heavy; if equal -> 3 light. If 5 lighter impossible. Good.

Could there be confusion: In case 3, first right pan heavier means left pan lighter. We say "If right pan heavier in Weighing 1". Good.

Need maybe include "H" and "L" definitions. Add: "H means heavy, L means light." Good.

Need ensure no commentary outside tags. Final only tags. Good.

Let's consider if user expects "exactly three times" maybe if branch identifies after two? Our tree always specifies third. Good.

Could there be issue with "identical-looking balls" and "exactly one counterfeit". Good.

Let's think if there is any subtle problem with using ball 9 in second weighing in unequal case: We know 9 genuine. Yes.

Let's maybe include a note: In Weighing 2 for unequal cases, ball 9 is known genuine. In equal case, balls 1-8 known genuine. Good.

Need maybe include "If the scale balances" etc. Good.

Let's ensure final answer is self-contained. It is.

Potential improvement: Use a table for decision tree? Bullets okay.

Let's verify all 12 possibilities:
1H: W1 L>R; W2 L>R; W3 1>2 -> 1H.
2H: W1 L>R; W2 L>R; W3 2>1 -> 2H.
3H: W1 L>R; W2 R>L; W3 5=9 -> 3H.
4H: W1 L>R; W2 =; W3 7=8 -> 4H.
5L: W1 L>R; W2 R>L; W3 5<9 -> 5L.
6L: W1 L>R; W2 L>R; W3 1=2 -> 6L.
7L: W1 L>R; W2 =; W3 7<8 -> 7L.
8L: W1 L>R; W2 =; W3 8<7 -> 8L.
1L: W1 R>L; W2 R>L; W3 1<2 -> 1L.
2L: W1 R>L; W2 R>L; W3 2<1 -> 2L.
3L: W1 R>L; W2 L>R; W3 5=9 -> 3L.
4L: W1 R>L; W2 =; W3 7=8 -> 4L.
5H: W1 R>L; W2 L>R; W3 5>9 -> 5H.
6H: W1 R>L; W2 R>L; W3 1=2 -> 6H.
7H: W1 R>L; W2 =; W3 7>8 -> 7H.
8H: W1 R>L; W2 =; W3 8>7 -> 8H.
9H: W1 =; W2 L>R; W3 9>10 -> 9H.
10H: W1 =; W2 L>R; W3 10>9 -> 10H.
11H: W1 =; W2 L>R; W3 9=10 -> 11H.
12H: W1 =; W2 =; W3 12>1 -> 12H.
9L: W1 =; W2 L<R; W3 9<10 -> 9L.
10L: W1 =; W2 L<R; W3 10<9 -> 10L.
11L: W1 =; W2 L<R; W3 9=10 -> 11L.
12L: W1 =; W2 =; W3 12<1 -> 12L.
All good.

Need maybe mention "left pan heavier" vs "right pan heavier" consistently. In branch 2c, "If right pan heavier" after W2. In branch 3b, "If left pan heavier". Good.

Could there be a problem in branch 2a: W3 7 vs 8. If 7 lighter, 7 light; if 8 lighter, 8 light. But if 4 heavy, both 7 and 8 genuine, equal. Good.

Branch 3a: W3 7 vs 8. If 7 heavier, 7 heavy; if 8 heavier, 8 heavy; if equal, 4 light. Good.

Need maybe include "The scale tells which side is heavier or equal". Good.

Let's think about final formatting. The user requested "Think first if you need to, then put your complete final answer between  tags. Everything inside those tags must be the answer itself, with no commentary about how you produced it." So final should not include analysis. Just answer. We can include the decision tree. Good.

One more thought: The phrase "You may use it exactly three times" could be interpreted as must use exactly three, not at most. Our strategy uses exactly three in all branches because we specify third. Good.

Could there be a requirement to "always identifies the counterfeit ball AND determines whether it is heavy or light". Yes.

Let's maybe include a small legend:
- "H" = heavy, "L" = light.
- "left" and "right" refer to pans.

Need ensure no accidental outside tags. Final:


No extra. Good.

Let's consider if there is any issue with using "If 5 heavier, impossible" inside answer. It is part of decision tree? It's okay. But maybe user doesn't want impossible branches. It's fine. Could omit to keep clean. But "full decision tree" maybe include all possible outcomes. I'll include parenthetical impossible. It doesn't hurt.

Could there be a branch where third weighing result "equal" in B3 identifies 3H, but we don't know if 3 is heavy? Yes we know from branch. Good.

Let's maybe make the tree more explicit with "Weighing 3" in each leaf. Good.

Potential alternative: In case first equal, second 9,10,11 vs 1,2,3. If left heavier, one of 9,10,11 heavy. Third 9 vs 10. If equal, 11 heavy. Good. If left lighter, one light. Third 9 vs 10. If equal, 11 light. Good.

Need maybe mention in case first equal, if W2 left heavier, we know 9-11 heavy not light because compared to known good. Good.

Let's think if there is any hidden assumption about balance scale comparing groups of equal number? It compares two groups; can weigh unequal numbers? Usually balance scale can compare any groups, but if unequal numbers, weight difference includes number difference. In our strategy all weighings have equal number of balls on each side (4 vs 4, 3 vs 3, 1 vs 1). Good.

Need maybe mention all weighings have equal counts, so no issue.

Let's ensure no arithmetic mistakes in branch labels:
Case 2: W1 left heavier. W2: 1,2,5 vs 3,6,9.
- If W2 left heavier: {1H,2H,6L}. W3 1 vs 2. If equal -> 6L. If 1 heavier -> 1H. If 2 heavier -> 2H. Good.
- If W2 right heavier: {3H,5L}. W3 5 vs 9. If equal -> 3H. If 5 lighter -> 5L. Good.
- If W2 equal: {4H,7L,8L}. W3 7 vs 8. If equal -> 4H. If 7 lighter -> 7L. If 8 lighter -> 8L. Good.

Case 3: W1 right heavier. W2 same.
- If W2 left heavier: {5H,3L}. W3 5 vs 9. If 5 heavier -> 5H. If equal -> 3L. Good.
- If W2 right heavier: {1L,2L,6H}. W3 1 vs 2. If equal -> 6H. If 1 lighter -> 1L. If 2 lighter -> 2L. Good.
- If W2 equal: {4L,7H,8H}. W3 7 vs 8. If equal -> 4L. If 7 heavier -> 7H. If 8 heavier -> 8H. Good.

All set.

Need maybe include "No other outcomes are possible" but not necessary.

Let's produce final. Ensure no markdown outside? It's okay inside tags. Could use plain text. The final answer can be markdown.

Double-check user's exact: "Describe a strategy that always identifies the counterfeit ball AND determines whether it is heavy or light, in three weighings, no matter which ball is counterfeit.

Set out the full decision tree: state what you weigh first, and then what you weigh in every possible outcome branch.

Think first if you need to, then put your complete final answer between  tags. Everything inside those tags must be the answer itself, with no commentary about how you produced it."

So final should be only answer. Good.

Let's maybe include "Number the balls 1 to 12." Good.

One more nuance: In case first equal, W2 9,10,11 vs 1,2,3. If W2 left heavier, one of 9,10,11 heavy. But what if 12 is counterfeit? No, W2 equal branch. Good.

Could there be a problem if in W2 left heavier, third 9 vs 10: if 9 heavier, 9 heavy; if 10 heavier, 10 heavy; if equal, 11 heavy. Good. If 11 heavy, 9 and 10 genuine equal. Good.

All good.

Let's final.
</think>

</details>

**Answer (as scored):**

Number the balls 1 through 12. Use “left” and “right” to mean the two pans of the scale. Let “H” mean heavy and “L” mean light.

**Weighing 1:** Put balls 1, 2, 3, 4 on the left pan and balls 5, 6, 7, 8 on the right pan.

---

## Branch 1: Weighing 1 balances

Then balls 1–8 are genuine, and the counterfeit is one of 9, 10, 11, 12.

**Weighing 2:** Put balls 9, 10, 11 on the left pan and balls 1, 2, 3 on the right pan.

### 1A: Weighing 2 balances
Then ball 12 is counterfeit.

**Weighing 3:** Put ball 12 on the left pan and ball 1 on the right pan.

- If 12 is heavier, ball 12 is heavy.
- If 12 is lighter, ball 12 is light.

### 1B: Weighing 2 left pan heavier
Then one of 9, 10, 11 is heavy.

**Weighing 3:** Put ball 9 on the left pan and ball 10 on the right pan.

- If 9 is heavier, ball 9 is heavy.
- If 10 is heavier, ball 10 is heavy.
- If they balance, ball 11 is heavy.

### 1C: Weighing 2 left pan lighter
Then one of 9, 10, 11 is light.

**Weighing 3:** Put ball 9 on the left pan and ball 10 on the right pan.

- If 9 is lighter, ball 9 is light.
- If 10 is lighter, ball 10 is light.
- If they balance, ball 11 is light.

---

## Branch 2: Weighing 1 left pan heavier

Then the counterfeit is among balls 1–8, and the possibilities are:

1H, 2H, 3H, 4H, 5L, 6L, 7L, 8L.

Balls 9–12 are genuine.

**Weighing 2:** Put balls 1, 2, 5 on the left pan and balls 3, 6, 9 on the right pan.

### 2A: Weighing 2 balances
Then the possibilities are: 4H, 7L, 8L.

**Weighing 3:** Put ball 7 on the left pan and ball 8 on the right pan.

- If they balance, ball 4 is heavy.
- If 7 is lighter, ball 7 is light.
- If 8 is lighter, ball 8 is light.

### 2B: Weighing 2 left pan heavier
Then the possibilities are: 1H, 2H, 6L.

**Weighing 3:** Put ball 1 on the left pan and ball 2 on the right pan.

- If they balance, ball 6 is light.
- If 1 is heavier, ball 1 is heavy.
- If 2 is heavier, ball 2 is heavy.

### 2C: Weighing 2 right pan heavier
Then the possibilities are: 3H, 5L.

**Weighing 3:** Put ball 5 on the left pan and ball 9 on the right pan.

- If they balance, ball 3 is heavy.
- If 5 is lighter, ball 5 is light.
- If 5 is heavier, this outcome is impossible under the stated assumptions.

---

## Branch 3: Weighing 1 right pan heavier

Then the counterfeit is among balls 1–8, and the possibilities are:

1L, 2L, 3L, 4L, 5H, 6H, 7H, 8H.

Balls 9–12 are genuine.

**Weighing 2:** Put balls 1, 2, 5 on the left pan and balls 3, 6, 9 on the right pan.

### 3A: Weighing 2 balances
Then the possibilities are: 4L, 7H, 8H.

**Weighing 3:** Put ball 7 on the left pan and ball 8 on the right pan.

- If they balance, ball 4 is light.
- If 7 is heavier, ball 7 is heavy.
- If 8 is heavier, ball 8 is heavy.

### 3B: Weighing 2 left pan heavier
Then the possibilities are: 5H, 3L.

**Weighing 3:** Put ball 5 on the left pan and ball 9 on the right pan.

- If 5 is heavier, ball 5 is heavy.
- If they balance, ball 3 is light.
- If 5 is lighter, this outcome is impossible under the stated assumptions.

### 3C: Weighing 2 right pan heavier
Then the possibilities are: 1L, 2L, 6H.

**Weighing 3:** Put ball 1 on the left pan and ball 2 on the right pan.

- If they balance, ball 6 is heavy.
- If 1 is lighter, ball 1 is light.
- If 2 is lighter, ball 2 is light.

