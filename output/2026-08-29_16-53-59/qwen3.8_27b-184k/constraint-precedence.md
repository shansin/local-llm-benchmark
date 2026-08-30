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

- Score: 1.0/10 — The response uses the forbidden word 'gear' and fails the precedence rule; it is not a valid summary under the given constraints.
- Tokens/s: 105.6
- Prefill tok/s: 1040.6
- TTFT: 16.57s
- Gen Time: 17.41s
- Output Tokens: 1811
- answer read from `answer_tag`

<details><summary>Reasoning (5150 chars, not scored)</summary>

Let me work through this carefully.

Requirements in order of precedence:
1. Exactly 3 sentences.
2. Every sentence must be at least 25 words long.
3. Must not exceed 60 words in total.
4. Do not use "chain" or "gear".
5. Name at least four distinct mechanical components.

Wait - requirements 2 and 3 conflict. If every sentence must be at least 25 words, then 3 sentences × 25 words = 75 words minimum. But requirement 3 says not to exceed 60 words total. These cannot both be met.

Since requirement 2 has lower number than requirement 3, requirement 2 wins. So I need every sentence to be at least 25 words, and I can exceed 60 words.

Let me also check: requirement 1 (exactly 3 sentences) and requirement 2 (each at least 25 words) - these can coexist.

So my constraints are:
- Exactly 3 sentences
- Each sentence at least 25 words
- Total can exceed 60 words (since req 2 beats req 3)
- No "chain" or "gear"
- At least four distinct mechanical components named

Let me think about mechanical components of a derailleur system:
- Derailleur (the mechanism itself)
- Cassette (the cluster of sprockets)
- Sprocket
- Cable
- Pulley
- Jockey wheel
- Cage
- Shifter
- Chainring
- Chain - CAN'T USE
- Gear - CAN'T USE

Let me pick: derailleur, cassette, sprocket, cable, pulley, shifter, cage, jockey wheel

Let me draft sentences:

Sentence 1: "When the rider operates the shifter, a cable pulls the derailleur mechanism laterally, guiding the chainring-adjacent drive loop onto a different sprocket of the cassette."

Wait, I can't use "chain" - but "drive loop" is a workaround. Actually let me re-read: "Do not use the words 'chain' or 'gear'." So I just can't use those exact words.

Let me try again:

Sentence 1: "When the rider operates the shifter, a tensioned cable pulls the derailleur mechanism laterally, guiding the drive loop onto a different sprocket within the cassette cluster."

Let me count: When(1) the(2) rider(3) operates(4) the(5) shifter(6) a(7) tensioned(8) cable(9) pulls(10) the(11) derailleur(12) mechanism(13) laterally(14) guiding(15) the(16) drive(17) loop(18) onto(19) a(20) different(21) sprocket(22) within(23) the(24) cassette(25) cluster(26). = 26 words. Good, ≥25.

Sentence 2: "The derailleur's cage, which houses two pulleys known as jockey wheels, maintains proper tension while the entire assembly shifts the drive path across the sprockets."

Let me count: The(1) derailleur's(2) cage(3) which(4) houses(5) two(6) pulleys(7) known(8) as(9) jockey(10) wheels(11) maintains(12) proper(13) tension(14) while(15) the(16) entire(17) assembly(18) shifts(19) the(20) drive(21) path(22) across(23) the(24) sprockets(25). = 25 words. Good, ≥25.

Sentence 3: "This lateral movement of the derailleur relative to the cassette effectively changes the mechanical advantage, allowing the rider to select a lighter or heavier resistance ratio."

Let me count: This(1) lateral(2) movement(3) of(4) the(5) derailleur(6) relative(7) to(8) the(9) cassette(10) effectively(11) changes(12) the(13) mechanical(14) advantage(15) allowing(16) the(17) rider(18) to(19) select(20) a(21) lighter(22) or(23) heavier(24) resistance(25) ratio(26). = 26 words. Good, ≥25.

Now let me check for "chain" or "gear" - I don't see either. Good.

Distinct mechanical components named:
1. Shifter
2. Cable
3. Derailleur
4. Sprocket
5. Cassette
6. Cage
7. Pulleys / jockey wheels

That's well over four. Good.

Let me verify no "chain" or "gear" appears:
- Sentence 1: no
- Sentence 2: no
- Sentence 3: no

Total words: 26 + 25 + 26 = 77 words. This exceeds 60, but requirement 2 (each sentence ≥25 words) takes precedence over requirement 3 (≤60 words total), so this is correct.

Let me double-check my word counts more carefully:

Sentence 1: "When the rider operates the shifter, a tensioned cable pulls the derailleur mechanism laterally, guiding the drive loop onto a different sprocket within the cassette cluster."
When(1) the(2) rider(3) operates(4) the(5) shifter(6) a(7) tensioned(8) cable(9) pulls(10) the(11) derailleur(12) mechanism(13) laterally(14) guiding(15) the(16) drive(17) loop(18) onto(19) a(20) different(21) sprocket(22) within(23) the(24) cassette(25) cluster(26). = 26 ✓

Sentence 2: "The derailleur's cage, which houses two pulleys known as jockey wheels, maintains proper tension while the entire assembly shifts the drive path across the sprockets."
The(1) derailleur's(2) cage(3) which(4) houses(5) two(6) pulleys(7) known(8) as(9) jockey(10) wheels(11) maintains(12) proper(13) tension(14) while(15) the(16) entire(17) assembly(18) shifts(19) the(20) drive(21) path(22) across(23) the(24) sprockets(25). = 25 ✓

Sentence 3: "This lateral movement of the derailleur relative to the cassette effectively changes the mechanical advantage, allowing the rider to select a lighter or heavier resistance ratio."
This(1) lateral(2) movement(3) of(4) the(5) derailleur(6) relative(7) to(8) the(9) cassette(10) effectively(11) changes(12) the(13) mechanical(14) advantage(15) allowing(16) the(17) rider(18) to(19) select(20) a(21) lighter(22) or(23) heavier(24) resistance(25) ratio(26). = 26 ✓

All good. Let me finalize.

</details>

**Answer (as scored):**

When the rider operates the shifter, a tensioned cable pulls the derailleur mechanism laterally, guiding the drive loop onto a different sprocket within the cassette cluster. The derailleur's cage, which houses two pulleys known as jockey wheels, maintains proper tension while the entire assembly shifts the drive path across the sprockets. This lateral movement of the derailleur relative to the cassette effectively changes the mechanical advantage, allowing the rider to select a lighter or heavier resistance ratio.

