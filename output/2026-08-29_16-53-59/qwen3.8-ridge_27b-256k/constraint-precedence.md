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

- Score: 1.0/10 — The response is not a summary of how a bicycle derailleur changes gear.
- Tokens/s: 56.2
- Prefill tok/s: 1188.5
- TTFT: 0.30s
- Gen Time: 51.61s
- Output Tokens: 2882
- answer read from `answer_tag`

<details><summary>Reasoning (9692 chars, not scored)</summary>

We need answer user's request. Need produce final only summary between . Need satisfy constraints with precedence: 1 exactly 3 sentences. 2 every sentence at least 25 words long. 3 total <=60 words. 4 no words "chain" or "gear". 5 name at least four distinct mechanical components.

This is impossible? Let's analyze. Exactly 3 sentences, each at least 25 words => total at least 75 words. Requirement 3 total <=60 words. Precedence lower number wins: req1 > req2 > req3. If cannot both meet req2 and req3, req2 wins. So we must have 3 sentences each >=25 words, total will be >=75, violating req3 but req3 lower precedence so okay. Need exactly 3 sentences. Need no chain/gear. Need at least four distinct mechanical components. Need output only summary. Need think first? User says Think first if need, then put final answer between tags. We can think here. Need ensure sentence count exactly 3. Need each sentence at least 25 words. Need total maybe 75+ words. Need no "chain" or "gear" as words. Avoid substring? Requirement says words "chain" or "gear". We can avoid. Need name at least four distinct mechanical components. Could be derailleur, cable, pulley, sprocket, shifter, cage, guide, jockey wheel, cassette, derailleur hanger, bolt, spring, pivot. Need summary of how bicycle derailleur changes gear. But cannot use gear. Can say changes speed ratio? Need no gear word. Need mechanical components. Need 3 sentences each >=25 words. Let's craft 3 sentences, count words carefully.

Sentence 1: "The shifter moves the cable, which pulls the derailleur cage and pivots the guide pulley toward the larger sprocket." Count: The(1) shifter2 moves3 the4 cable5 which6 pulls7 the8 derailleur9 cage10 and11 pivots12 the13 guide14 pulley15 toward16 the17 larger18 sprocket19. Only 19. Need >=25. Add words. "The shifter moves the cable, which pulls the derailleur cage and pivots the guide pulley toward the larger sprocket on the cassette." Count: The1 shifter2 moves3 the4 cable5 which6 pulls7 the8 derailleur9 cage10 and11 pivots12 the13 guide14 pulley15 toward16 the17 larger18 sprocket19 on20 the21 cassette22. 22. Need 25. Add "carefully" etc. "The shifter moves the cable, which pulls the derailleur cage and pivots the guide pulley carefully toward the larger sprocket on the cassette." Count: The1 shifter2 moves3 the4 cable5 which6 pulls7 the8 derailleur9 cage10 and11 pivots12 the13 guide14 pulley15 carefully16 toward17 the18 larger19 sprocket20 on21 the22 cassette23. 23. Need 25. Add "smoothly" maybe. "The shifter moves the cable, which pulls the derailleur cage and pivots the guide pulley carefully and smoothly toward the larger sprocket on the cassette." Count: The1 shifter2 moves3 the4 cable5 which6 pulls7 the8 derailleur9 cage10 and11 pivots12 the13 guide14 pulley15 carefully16 and17 smoothly18 toward19 the20 larger21 sprocket22 on23 the24 cassette25. Good 25.

Sentence 2: Need >=25. "As the cable tension changes, the spring returns the derailleur, while the jockey wheel keeps the drive band under proper tension." Count: As1 the2 cable3 tension4 changes5 the6 spring7 returns8 the9 derailleur10 while11 the12 jockey13 wheel14 keeps15 the16 drive17 band18 under19 proper20 tension21. 21. Need 25. Add "consistently" etc. "As the cable tension changes, the spring returns the derailleur, while the jockey wheel keeps the drive band under proper and consistent tension." Count: As1 the2 cable3 tension4 changes5 the6 spring7 returns8 the9 derailleur10 while11 the12 jockey13 wheel14 keeps15 the16 drive17 band18 under19 proper20 and21 consistent22 tension23. 23. Need 25. Add "mechanically"? "As the cable tension changes, the spring returns the derailleur, while the jockey wheel keeps the drive band under proper and consistent mechanical tension." Count: As1 the2 cable3 tension4 changes5 the6 spring7 returns8 the9 derailleur10 while11 the12 jockey13 wheel14 keeps15 the16 drive17 band18 under19 proper20 and21 consistent22 mechanical23 tension24. 24. Need 25. Add "overall"? "As the cable tension changes, the spring returns the derailleur, while the jockey wheel keeps the drive band under proper and consistent overall mechanical tension." Count: As1 the2 cable3 tension4 changes5 the6 spring7 returns8 the9 derailleur10 while11 the12 jockey13 wheel14 keeps15 the16 drive17 band18 under19 proper20 and21 consistent22 overall23 mechanical24 tension25. Good. But "drive band" maybe not component? It's okay. Need distinct components: shifter, cable, derailleur cage, guide pulley, sprocket, cassette, spring, jockey wheel. At least four.

Sentence 3: Need >=25. "This lateral movement transfers the drive band to another sprocket, altering the mechanical advantage without stopping the rider." Count: This1 lateral2 movement3 transfers4 the5 drive6 band7 to8 another9 sprocket10 altering11 the12 mechanical13 advantage14 without15 stopping16 the17 rider18. 18. Need 25. Add words. "This lateral movement transfers the drive band to another sprocket, altering the mechanical advantage without stopping the rider or interrupting pedaling effort." Count: This1 lateral2 movement3 transfers4 the5 drive6 band7 to8 another9 sprocket10 altering11 the12 mechanical13 advantage14 without15 stopping16 the17 rider18 or19 interrupting20 pedaling21 effort22. 22. Need 25. Add "smoothly" etc. "This lateral movement transfers the drive band to another sprocket, altering the mechanical advantage without stopping the rider or interrupting the pedaling effort." Count: This1 lateral2 movement3 transfers4 the5 drive6 band7 to8 another9 sprocket10 altering11 the12 mechanical13 advantage14 without15 stopping16 the17 rider18 or19 interrupting20 the21 pedaling22 effort23. 23. Need 25. Add "continuously"? "This lateral movement transfers the drive band to another sprocket, altering the mechanical advantage without stopping the rider or interrupting the continuous pedaling effort." Count: This1 lateral2 movement3 transfers4 the5 drive6 band7 to8 another9 sprocket10 altering11 the12 mechanical13 advantage14 without15 stopping16 the17 rider18 or19 interrupting20 the21 continuous22 pedaling23 effort24. 24. Need 25. Add "smoothly" before altering? "This lateral movement smoothly transfers the drive band to another sprocket, altering the mechanical advantage without stopping the rider or interrupting the continuous pedaling effort." Count: This1 lateral2 movement3 smoothly4 transfers5 the6 drive7 band8 to9 another10 sprocket11 altering12 the13 mechanical14 advantage15 without16 stopping17 the18 rider19 or20 interrupting21 the22 continuous23 pedaling24 effort25. Good.

Need check no "chain" or "gear". We used drive band, sprocket, cassette, derailleur, shifter, cable, pulley, spring, jockey wheel. No chain/gear. Need exactly 3 sentences. Punctuation: Sentence 1 ends period. Sentence 2 ends period. Sentence 3 ends period. Need ensure no abbreviations with periods. Good.

Need total words? Req3 lower precedence, but we can maybe try to satisfy? Impossible with req2. But maybe if each sentence at least 25 words, total 75. We have 25+25+25=75. Requirement 3 cannot be met. Precedence says req2 wins over req3. So okay. But maybe they expect we drop req3. Need not mention.

Need verify word counts exactly? Let's count with spaces. Sentence1: "The shifter moves the cable, which pulls the derailleur cage and pivots the guide pulley carefully and smoothly toward the larger sprocket on the cassette."
Words: The(1) shifter(2) moves(3) the(4) cable,(5) which(6) pulls(7) the(8) derailleur(9) cage(10) and(11) pivots(12) the(13) guide(14) pulley(15) carefully(16) and(17) smoothly(18) toward(19) the(20) larger(21) sprocket(22) on(23) the(24) cassette.(25). Good.

Sentence2: "As the cable tension changes, the spring returns the derailleur, while the jockey wheel keeps the drive band under proper and consistent overall mechanical tension."
Words: As1 the2 cable3 tension4 changes,5 the6 spring7 returns8 the9 derailleur,10 while11 the12 jockey13 wheel14 keeps15 the16 drive17 band18 under19 proper20 and21 consistent22 overall23 mechanical24 tension.25. Good.

Sentence3: "This lateral movement smoothly transfers the drive band to another sprocket, altering the mechanical advantage without stopping the rider or interrupting the continuous pedaling effort."
Words: This1 lateral2 movement3 smoothly4 transfers5 the6 drive7 band8 to9 another10 sprocket,11 altering12 the13 mechanical14 advantage15 without16 stopping17 the18 rider19 or20 interrupting21 the22 continuous23 pedaling24 effort.25. Good.

Need at least four distinct mechanical components: shifter, cable, derailleur cage, guide pulley, sprocket, cassette, spring, jockey wheel. Yes.

Potential issue: "derailleur cage" maybe one component? "guide pulley" maybe component. "jockey wheel" component. "spring" component. Good.

Potential issue: "drive band" could be considered chain? But not word chain. It's okay. Could use "drive band" to avoid chain. But is it a mechanical component? It's the chain, but not named. Requirement 5 only at least four components. We have enough.

Potential issue: "gear" not used. Good.

Potential issue: Sentence 2 "the spring returns the derailleur" maybe not accurate? Derailleur spring returns? It's okay summary.

Need output only summary between tags. No extra. Ensure tags exactly? User: put complete final answer between  tags. Everything inside tags must be answer itself. So final:


Need maybe no newline? Fine.

Double-check sentence count: 3 sentences. Punctuation: comma doesn't count. Periods only three. Good.

Could there be issue with "cable" as mechanical component? yes.

Need maybe "derailleur" itself component. Good.

Now final.
</think>

</details>

**Answer (as scored):**

The shifter moves the cable, which pulls the derailleur cage and pivots the guide pulley carefully and smoothly toward the larger sprocket on the cassette. As the cable tension changes, the spring returns the derailleur, while the jockey wheel keeps the drive band under proper and consistent overall mechanical tension. This lateral movement smoothly transfers the drive band to another sprocket, altering the mechanical advantage without stopping the rider or interrupting the continuous pedaling effort.

