# instruction / constraint-precedence

**Prompt:** Write a summary of how a bicycle derailleur changes gear.

Requirements, in order of precedence — where two requirements cannot both be met, the one
with the lower number wins, and you must follow it:

1. The summary must be exactly 3 sentences.
2. Every sentence must be at least 25 words long.
3. The summary must not exceed 60 words in total.
4. Do not use the words "chain" or "gear".
5. Name at least four distinct mechanical components.

Output only the summary. Do not explain your choices, do not mention the requirements, and
do not note which ones you dropped.

## Repeat 1 (seed 0)

- Score: 2.5/10 — Response uses forbidden word 'gear' and fails requirement 4; also uses 'chain' concept via 'metal loop of links' but main failure is forbidden word, and precedence not correctly applied per criteria.
- Tokens/s: 208.5
- Prefill tok/s: 2305.5
- TTFT: 0.13s
- Gen Time: 12.43s
- Output Tokens: 2565
- answer read from `answer_tag`

<details><summary>Reasoning (8288 chars, not scored)</summary>

Let me analyze the requirements carefully.

Requirements in order of precedence:
1. Exactly 3 sentences.
2. Every sentence must be at least 25 words long.
3. Must not exceed 60 words in total.
4. Don't use "chain" or "gear".
5. Name at least four distinct mechanical components.

Now, there's a conflict. Requirement 1 says exactly 3 sentences. Requirement 2 says every sentence must be at least 25 words. So 3 sentences × 25 words = at least 75 words. But requirement 3 says must not exceed 60 words total.

These conflict. Requirement 1 has higher precedence than requirement 3. So I must follow requirement 1 (exactly 3 sentences) over requirement 3 (max 60 words).

Wait, let me re-read: "where two requirements cannot both be met, the one with the lower number wins."

So requirement 1 (lower number) wins over requirement 3. So I must have exactly 3 sentences.

Requirement 2 (every sentence at least 25 words) — this conflicts with requirement 3 (max 60 words). Requirement 2 has higher precedence than requirement 3. So I follow requirement 2 over requirement 3.

So the conflict is between requirement 2 and requirement 3. Requirement 2 wins. So every sentence must be at least 25 words.

But wait, can I meet requirement 1 and requirement 2 together? 3 sentences × 25 words = 75 words minimum. That's fine for requirement 1 and 2.

But requirement 3 (max 60 words) conflicts with both. Since requirement 1 and 2 both have higher precedence than requirement 3, I drop requirement 3.

So I need:
- Exactly 3 sentences (requirement 1)
- Every sentence at least 25 words (requirement 2)
- Don't use "chain" or "gear" (requirement 4)
- Name at least four distinct mechanical components (requirement 5)

I can't meet requirement 3 (max 60 words) because it conflicts with 1 and 2.

So my summary should be 3 sentences, each at least 25 words, no "chain" or "gear", naming at least 4 mechanical components.

Let me think about the mechanical components of a derailleur:
- Derailleur (the mechanism)
- Cage / jockey wheels / pulleys
- Cable
- Derailleur hanger
- Cassette / cogs
- Shift lever / shifter
- Frame

Let me write about how a derailleur changes gear without using "chain" or "gear".

Let me draft:

Sentence 1: "When you operate the shift lever, tension on the control cable changes, causing the rear derailleur body to swing and reposition the two pulley wheels within its cage."

Let me count words: When(1) you(2) operate(3) the(4) shift(5) lever,(6) tension(7) on(8) the(9) control(10) cable(11), changes,(12) causing(13) the(14) rear(15) derailleur(16) body(17) to(18) swing(19) and(20) reposition(21) the(22) two(23) pulley(24) wheels(25) within(26) its(27) cage.(28)

28 words. Good, at least 25.

Sentence 2: "This lateral movement pushes the metal loop of links sideways, guiding it onto a differently sized cog and thereby altering the rotational ratio delivered to the rear wheel."

Count: This(1) lateral(2) movement(3) pushes(4) the(5) metal(6) loop(7) of(8) links(9) sideways,(10) guiding(11) it(12) onto(13) a(14) differently(15) sized(16) cog(17) and(18) thereby(19) altering(20) the(21) rotational(22) ratio(23) delivered(24) to(25) the(26) rear(27) wheel.(28)

28 words. Good.

Wait, I used "metal loop of links" to avoid "chain". That's fine.

Sentence 3: "Selecting a larger cog produces easier pedaling for climbing, while a smaller cog increases speed, so the rider continuously adjusts the mechanism to match terrain and effort."

Count: Selecting(1) a(2) larger(3) cog(4) produces(5) easier(6) pedaling(7) for(8) climbing,(9) while(10) a(11) smaller(12) cog(13) increases(14) speed,(15) so(16) the(17) rider(18) continuously(19) adjusts(20) the(21) mechanism(22) to(23) match(24) terrain(25) and(26) effort.(27)

27 words. Good.

Mechanical components named: shift lever, control cable, derailleur body, pulley wheels, cage, cog, rear wheel, mechanism. That's more than four distinct mechanical components.

Let me verify no "chain" or "gear" used. I used "metal loop of links" instead of chain. No "gear". Good.

Let me double check the word count. Total: 28 + 28 + 27 = 83 words. That exceeds 60, but requirement 3 is dropped due to precedence.

Actually, let me reconsider. The requirement says "Do not use the words 'chain' or 'gear'." I need to make sure I don't accidentally use them. Let me check my sentences again.

Sentence 1: "When you operate the shift lever, tension on the control cable changes, causing the rear derailleur body to swing and reposition the two pulley wheels within its cage." — no chain, no gear.

Sentence 2: "This lateral movement pushes the metal loop of links sideways, guiding it onto a differently sized cog and thereby altering the rotational ratio delivered to the rear wheel." — no chain, no gear.

Sentence 3: "Selecting a larger cog produces easier pedaling for climbing, while a smaller cog increases speed, so the rider continuously adjusts the mechanism to match terrain and effort." — no chain, no gear.

Good.

Now let me reconsider the precedence logic once more. The requirements are in order of precedence where lower number wins when they conflict.

Requirement 1: exactly 3 sentences.
Requirement 2: every sentence at least 25 words.
Requirement 3: max 60 words total.

If I have 3 sentences each at least 25 words, that's at least 75 words. This conflicts with requirement 3 (max 60). 

Requirement 1 (lower number) wins over requirement 3. Requirement 2 (lower number) wins over requirement 3. So I keep requirements 1 and 2, drop requirement 3.

So my answer with 83 words is correct.

Actually, wait. Let me reconsider whether I can satisfy all of 1, 2, 3 simultaneously. 

3 sentences, each ≥25 words → ≥75 words. But max 60 words. Impossible. So at least one of 1, 2, 3 must be dropped. Since 1 and 2 have higher precedence than 3, drop 3.

So the answer is 3 sentences, each ≥25 words, no chain/gear, ≥4 mechanical components.

My draft satisfies all of these. Let me finalize.

Actually, let me reconsider the mechanical components more carefully. The requirement says "Name at least four distinct mechanical components." Let me make sure I'm naming actual mechanical components clearly.

- shift lever ✓
- control cable ✓
- derailleur body ✓
- pulley wheels (jockey wheels) ✓
- cage ✓
- cog ✓
- rear wheel ✓

That's plenty. Good.

Let me refine the sentences to be clear and accurate about how a derailleur works.

Actually, the description is accurate. When you shift, the cable tension changes, the derailleur moves, pushing the chain (which I call "metal loop of links") onto different cogs, changing the gear ratio.

Let me finalize my answer.

Final answer:

"When you operate the shift lever, tension on the control cable changes, causing the rear derailleur body to swing and reposition the two pulley wheels within its cage. This lateral movement pushes the metal loop of links sideways, guiding it onto a differently sized cog and thereby altering the rotational ratio delivered to the rear wheel. Selecting a larger cog produces easier pedaling for climbing, while a smaller cog increases speed, so the rider continuously adjusts the mechanism to match terrain and effort."

Let me recount each sentence.

Sentence 1: When(1) you(2) operate(3) the(4) shift(5) lever,(6) tension(7) on(8) the(9) control(10) cable(11), changes,(12) causing(13) the(14) rear(15) derailleur(16) body(17) to(18) swing(19) and(20) reposition(21) the(22) two(23) pulley(24) wheels(25) within(26) its(27) cage.(28) = 28 words ✓

Sentence 2: This(1) lateral(2) movement(3) pushes(4) the(5) metal(6) loop(7) of(8) links(9) sideways,(10) guiding(11) it(12) onto(13) a(14) differently(15) sized(16) cog(17) and(18) thereby(19) altering(20) the(21) rotational(22) ratio(23) delivered(24) to(25) the(26) rear(27) wheel.(28) = 28 words ✓

Sentence 3: Selecting(1) a(2) larger(3) cog(4) produces(5) easier(6) pedaling(7) for(8) climbing,(9) while(10) a(11) smaller(12) cog(13) increases(14) speed,(15) so(16) the(17) rider(18) continuously(19) adjusts(20) the(21) mechanism(22) to(23) match(24) terrain(25) and(26) effort.(27) = 27 words ✓

All sentences ≥25 words. 3 sentences. No chain/gear. 4+ mechanical components. 

Total 83 words. Requirement 3 dropped due to precedence.

This is my final answer.
</think>

</details>

**Answer (as scored):**

When you operate the shift lever, tension on the control cable changes, causing the rear derailleur body to swing and reposition the two pulley wheels within its cage. This lateral movement pushes the metal loop of links sideways, guiding it onto a differently sized cog and thereby altering the rotational ratio delivered to the rear wheel. Selecting a larger cog produces easier pedaling for climbing, while a smaller cog increases speed, so the rider continuously adjusts the mechanism to match terrain and effort.

