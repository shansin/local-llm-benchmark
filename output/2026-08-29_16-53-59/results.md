# Benchmark Results — 2026-08-29_16-53-59

## System Info

- **CPU:** AMD Ryzen 9 9950X3D 16-Core Processor
- **RAM:** 59.2 GB
- **GPUs:** NVIDIA GeForce RTX 5070 Ti, 16303 MiB; NVIDIA GeForce RTX 5070 Ti, 16303 MiB
- **Ollama GPUs:** CUDA_VISIBLE_DEVICES=all
- **Generation params:** `temperature=0.0, top_p=1.0, top_k=40, seed=0, num_ctx=131072, num_predict=-1`
- **Tasks:** 43 across 9 categories
- **Total Benchmark Runtime:** 4h 50m 10s

## Performance (fixed probe)

| Model | Params | Quant | Tokens/s | IQR | Prefill tok/s | TTFT p50 | TTFT p90 | Cold load (s) | VRAM (MiB) |
|-------|--------|-------|----------|-----|---------------|----------|----------|---------------|------------|
| qwen3.8-ridge:27b-256k | 27.3B | IQ2_M | 56.7 | ±0.1 | 1374.5 | 0.53 | 0.54 | 0.0 | 17727 |
| nemotron-3.5-lightning:30b-208k | 32.9B | Q4_K_M | 293.4 | ±32.2 | 3274.5 | 0.18 | 0.19 | 13.9 | 25839 |
| ornith-1.5:35b-240k | 35.5B | Q4_K_M | 209.4 | ±0.7 | 3562.9 | 0.20 | 0.20 | 10.2 | 23244 |
| qwen3.8:27b-184k | 27.3B | Q4_K_M | 86.1 | ±5.8 | 1264.7 | 2.78 | 2.92 | 9.9 | 20883 |

## Prefill scaling (tok/s by input length)

Measured input tokens are reported by Ollama, so these are actual lengths, not targets. Lengths beyond the configured `num_ctx` are skipped rather than silently truncated.

| Model | ~512 tok | ~4096 tok | ~16384 tok |
|-------|----------|-----------|------------|
| qwen3.8-ridge:27b-256k | 1521 | 2438 | 2312 |
| nemotron-3.5-lightning:30b-208k | 4118 | 5153 | 4831 |
| ornith-1.5:35b-240k | 4359 | 7176 | 6915 |
| qwen3.8:27b-184k | 1423 | 1642 | 1467 |

## Answer completeness

Generations that produced nothing to score, and generations whose answer arrived buried in undelimited reasoning. Both distort scores in ways the quality table cannot show.

| Model | Generations | Truncated | …discarded | No answer | Leaked reasoning | Errors |
|-------|-------------|-----------|------------|-----------|------------------|--------|
| qwen3.8-ridge:27b-256k | 43 | 0 | 0 | 0 | 2 | 0 |
| nemotron-3.5-lightning:30b-208k | 43 | 0 | 0 | 0 | 0 | 3 |
| ornith-1.5:35b-240k | 43 | 2 | 0 | 0 | 1 | 0 |
| qwen3.8:27b-184k | 43 | 0 | 0 | 1 | 0 | 0 |

**Truncated** generations hit `num_ctx=131072` and stopped mid-thought. They are scored as missing answers, because that is what they are — not as wrong ones. Raise `NUM_CTX` and re-run before reading those rows as quality.

**Leaked reasoning** counts answers that open with first-person planning prose despite the instruction to put the final answer inside `<answer></answer>` — the model ignored the delimiter it was asked for. Deterministic checks on those responses may be measuring the planning notes rather than the answer, so read that model's objective scores with suspicion.

## Quality

Blended score: 60% deterministic checks, 40% judge (muse-glimmer:30b). Tasks with no checks are scored by the judge alone. Each prompt is sampled once at temperature 0, so treat small differences between models as ties.

| Model | Coding | Faithfulness | Instruction | Knowledge | Longcontext | Reasoning | Statetrack | Transformation | Writing | Avg Score |
|-------|--------|--------------|-------------|-----------|-------------|-----------|------------|----------------|---------|-----------|
| qwen3.8-ridge:27b-256k | 8.9 | 9.7 | 8.4 | 9.6 | 5.1 | 9.2 | 10.0 | 10.0 | 7.1 | **8.70** |
| nemotron-3.5-lightning:30b-208k | 9.8 | 10.0 | 8.8 | 9.6 | 6.2 | 7.3 | 6.8 | 10.0 | 7.6 | **8.63** |
| ornith-1.5:35b-240k | 8.0 | 10.0 | 9.0 | 9.6 | 9.4 | 8.3 | 8.7 | 9.8 | 7.7 | **8.80** |
| qwen3.8:27b-184k | 9.2 | 9.8 | 7.6 | 9.6 | 9.4 | 9.0 | 9.0 | 10.0 | 8.9 | **9.03** |

## Judge calibration

On tasks with verifiable checks, how far the judge's score sits from the measured one. A large positive number means the judge is scoring answers higher than they deserve — the single most useful thing to know about an LLM-as-judge setup.

| Model | Coding | Faithfulness | Instruction | Knowledge | Longcontext | Reasoning | Statetrack | Transformation | Writing |
|-------|--------|--------------|-------------|-----------|-------------|-----------|------------|----------------|---------|
| qwen3.8-ridge:27b-256k | +0.4 | -0.1 | -2.2 | -0.9 | +6.5 | -1.8 | +0.0 | +0.0 | -2.2 |
| nemotron-3.5-lightning:30b-208k | -0.2 | +0.0 | -3.0 | -1.1 | +1.0 | -1.1 | +0.3 | +0.0 | -3.5 |
| ornith-1.5:35b-240k | +1.1 | +0.0 | -2.6 | -0.9 | +0.7 | -4.6 | +1.0 | -0.5 | -0.7 |
| qwen3.8:27b-184k | +1.0 | +0.1 | -2.5 | -1.0 | +0.7 | -3.0 | +0.8 | +0.0 | -2.8 |

## Per-task scores

Each cell is `blended (objective / judge)` averaged over repeats.

| Category | Task | Difficulty | qwen3.8-ridge:27b-256k | nemotron-3.5-lightning:30b-208k | ornith-1.5:35b-240k | qwen3.8:27b-184k |
|----------|------|------------|------------------------|---------------------------------|---------------------|------------------|
| coding | balanced-brackets | easy | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 9.9 (10.0 / 9.8) | 10.0 (10.0 / 10.0) |
| coding | fix-date-overlap | hard | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| coding | fix-run-length | medium | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 9.5 (10.0 / 8.8) | 10.0 (10.0 / 10.0) |
| coding | fix-window-sum | medium | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| coding | lru-cache | hard | 2.0 (0.0 / 5.0) | 9.9 (10.0 / 9.8) | 4.0 (0.0 / 10.0) | 9.9 (10.0 / 9.8) |
| coding | merge-intervals | medium | 9.9 (10.0 / 9.8) | 9.9 (10.0 / 9.8) | 9.9 (10.0 / 9.8) | 3.4 (0.0 / 8.5) |
| coding | parse-logs | medium | 9.6 (10.0 / 9.0) | 9.9 (10.0 / 9.8) | 9.9 (10.0 / 9.8) | 9.9 (10.0 / 9.8) |
| coding | topological-sort | hard | 9.9 (10.0 / 9.8) | 8.6 (9.0 / 8.0) | 0.4 (0.0 / 1.0) | 10.0 (10.0 / 10.0) |
| faithfulness | false-premises | medium | 9.0 (9.1 / 8.8) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| faithfulness | overclaimed-evidence | hard | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| faithfulness | unanswerable-questions | hard | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 9.4 (9.3 / 9.5) |
| faithfulness | unusual-semantics | hard | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| instruction | chained-conversions | medium | 9.6 (10.0 / 9.0) | 9.3 (10.0 / 8.2) | 9.7 (10.0 / 9.2) | 9.7 (10.0 / 9.2) |
| instruction | constraint-precedence | hard | 6.4 (10.0 / 1.0) | 6.4 (10.0 / 1.0) | 7.0 (10.0 / 2.5) | 6.4 (10.0 / 1.0) |
| instruction | impossible-constraints | hard | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| instruction | lipogram | hard | 3.4 (5.0 / 1.0) | 9.6 (10.0 / 9.0) | 9.7 (10.0 / 9.2) | 0.4 (0.0 / 1.0) |
| instruction | negative-constraints | easy | 9.4 (10.0 / 8.5) | 6.4 (10.0 / 1.0) | 6.4 (10.0 / 1.0) | 6.4 (10.0 / 1.0) |
| instruction | nested-format | hard | 9.9 (10.0 / 9.8) | 9.9 (10.0 / 9.8) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| instruction | strict-json | medium | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| knowledge | antibiotic-resistance | medium | 9.7 (10.0 / 9.2) | 9.6 (10.0 / 9.0) | 9.7 (10.0 / 9.2) | 9.6 (10.0 / 9.0) |
| knowledge | attention-mechanism | medium | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) |
| knowledge | fission-fusion | medium | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) |
| knowledge | floating-point | hard | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) | 9.7 (10.0 / 9.2) | 9.6 (10.0 / 9.0) |
| knowledge | tcp-vs-udp | easy | 9.7 (10.0 / 9.2) | 9.4 (10.0 / 8.5) | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) |
| longcontext | ledger-audit | hard | 4.0 (0.0 / 10.0) | 0.4 (0.0 / 1.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| longcontext | policy-lookup | hard | 8.3 (7.5 / 9.5) | 8.3 (7.5 / 9.5) | 8.3 (7.5 / 9.5) | 8.3 (7.5 / 9.5) |
| longcontext | scattered-facts | hard | 3.0 (0.0 / 7.5) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| reasoning | knights-knaves | medium | 10.0 (10.0 / 10.0) | 9.9 (10.0 / 9.8) | 8.2 (10.0 / 5.5) | 10.0 (10.0 / 10.0) |
| reasoning | schedule-constraints | medium | 9.9 (10.0 / 9.8) | 9.7 (10.0 / 9.2) | 8.8 (10.0 / 7.0) | 9.6 (10.0 / 9.0) |
| reasoning | twelve-balls | hard | 9.0 | 8.0 | 8.8 | 9.8 |
| reasoning | work-rate | easy | 10.0 (10.0 / 10.0) | 8.3 (10.0 / 5.8) | 8.3 (10.0 / 5.8) | 8.3 (10.0 / 5.8) |
| reasoning | zebra-puzzle | medium | 7.2 (10.0 / 3.0) | 0.4 (0.0 / 1.0) | 7.3 (10.0 / 3.2) | 7.3 (10.0 / 3.2) |
| statetrack | ledger-balance | medium | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| statetrack | robot-grid | medium | 10.0 (10.0 / 10.0) | 0.4 (0.0 / 1.0) | 10.0 (10.0 / 10.0) | 6.9 (6.0 / 8.2) |
| statetrack | stack-machine | hard | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 6.2 (5.0 / 8.0) | 10.0 (10.0 / 10.0) |
| transformation | computed-table | medium | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| transformation | csv-to-json | medium | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 9.4 (10.0 / 8.5) | 10.0 (10.0 / 10.0) |
| transformation | schema-migration | hard | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) | 10.0 (10.0 / 10.0) |
| writing | dialogue-only | hard | 0.4 (0.0 / 1.0) | 6.4 (10.0 / 1.0) | 3.6 (0.0 / 9.0) | 6.4 (10.0 / 1.0) |
| writing | explain-to-child | easy | 9.4 (10.0 / 8.5) | 9.0 (10.0 / 7.5) | 9.5 (10.0 / 8.8) | 9.4 (10.0 / 8.5) |
| writing | lighthouse | medium | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) | 9.6 (10.0 / 9.0) |
| writing | technical-to-plain | medium | 9.8 (10.0 / 9.5) | 9.5 (10.0 / 8.8) | 9.5 (10.0 / 8.8) | 9.5 (10.0 / 8.8) |
| writing | unreliable-narrator | hard | 6.4 (10.0 / 1.0) | 3.4 (5.0 / 1.0) | 6.4 (10.0 / 1.0) | 9.6 (10.0 / 9.0) |

## Judge Details

### qwen3.8-ridge:27b-256k

- **coding/balanced-brackets:** 10.0 — Function is correctly named with type hints, uses stack to enforce nesting order, ignores non-brackets, handles empty string and mismatches, and all 9 tests passed.
    - `code_exec`: 9/9 tests passed
- **coding/fix-date-overlap:** 10.0 — The response fixes the off-by-one inclusive bug with +1, preserves ValueError and guard, and gives a concise two-sentence explanation.
    - `code_exec`: 9/9 tests passed
- **coding/fix-run-length:** 10.0 — The response fixes both the double-counting and missing final run bugs, keeps signature/docstring, passes all tests, and explains in two sentences.
    - `code_exec`: 9/9 tests passed
- **coding/fix-window-sum:** 10.0 — The response fixes both bugs by initializing best to the first window sum, preserves signature and ValueError, keeps sliding window O(n), and explanation is within three sentences.
    - `code_exec`: 9/9 tests passed
- **coding/lru-cache:** 5.0 — NameError for _Node makes code non-functional; otherwise design is correct with O(1) structures and type hints.
    - `code_exec`: 0/8 tests passed — first failure: test_get_and_put_round_trip: NameError: name '_Node' is not defined
- **coding/merge-intervals:** 9.8 — Correct O(n log n) merge with type hints, handles all edge cases, 5 tests pass, code is clear and runnable.
    - `code_exec`: 12/12 tests passed
- **coding/parse-logs:** 9.0 — Function meets spec with correct keys, counts, sorted services, first_error handling, type hints, and robust malformed handling; minor timestamp validation adds strictness but does not break tests.
    - `code_exec`: 9/9 tests passed
- **coding/topological-sort:** 9.8 — Implementation uses Kahn's algorithm O(V+E), includes all nodes, handles empty graph, has type hints, raises ValueError naming a node in cycle, and passes all automated tests.
    - `code_exec`: 10/10 tests passed
- **faithfulness/false-premises:** 8.8 — Premise flags and key facts are correct; q3 note gives a computed value despite false premise and q5 omits penicillin mention, slightly reducing accuracy/completeness.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1.premise_ok == False
    - `json_path`: q2.premise_ok == True
    - `json_path`: q3.premise_ok == False
    - `json_path`: q4.premise_ok == True
    - `json_path`: q5.premise_ok == False
    - `json_path`: q6.premise_ok == False
    - `contains_any`: found: photoelectric
    - `contains_any`: found: 687
    - `contains_any`: none of: penicillin
    - `contains_any`: found: Saturn
    - `contains_any`: found: tidal locking
- **faithfulness/overclaimed-evidence:** 10.0 — Response supplies two verbatim quotations supporting ahead of schedule and under budget, correctly marks claim not fully supported and identifies staff satisfaction as unsupported.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: count == 2
    - `json_path`: claim_fully_supported == False
    - `contains_all`: all present
    - `contains_any`: found: satisf
    - `regex`: forbidden pattern absent
- **faithfulness/unanswerable-questions:** 10.0 — All seven answers match the passage and required 'not stated' for unanswerable items, with valid JSON and no extra commentary.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1 == 14
    - `json_path`: q2 == 41
    - `json_path`: q3 == 9
    - `json_path`: q4 == 23
    - `json_path`: q5 == 'not stated'
    - `json_path`: q6 == 'not stated'
    - `json_path`: q7 == 'not stated'
    - `match_count`: 3 matches, as required
- **faithfulness/unusual-semantics:** 10.0 — All seven answers match the specification exactly and JSON is correctly formatted with no extra text.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1 == 20
    - `json_path`: q2 == 3
    - `json_path`: q3 == 2
    - `json_path`: q4 == -3
    - `json_path`: q5 == True
    - `json_path`: q6 == 1
    - `json_path`: q7 == 'error'
- **instruction/chained-conversions:** 9.0 — All conversions and computations are correct and clearly labeled in order, but ASCII art is only one line tall per character instead of at least 3 lines tall as required.
    - `contains_all`: all present
    - `contains_any`: found: F7, f7
- **instruction/constraint-precedence:** 1.0 — The response is not a summary of how a bicycle derailleur changes gear.
    - `word_count`: 75 words, within range
    - `regex`: forbidden pattern absent
    - `match_count`: 3 matches, as required
    - `regex`: forbidden pattern absent
- **instruction/impossible-constraints:** 10.0 — Response matches expected IMPOSSIBLE line naming requirements 2 and 3 with no extra content.
    - `regex`: pattern found
    - `contains_all`: all present
    - `regex`: forbidden pattern absent
    - `word_count`: 6 words, within range
- **instruction/lipogram:** 1.0 — The AI response contains the letter 'e' and is not a product description, violating the primary zero-e constraint and all formatting rules.
    - `regex`: forbidden pattern present: 'e'
    - `word_count`: 1921 words, within range
- **instruction/negative-constraints:** 8.5 — Plot is recognizable with constraints fully satisfied; completeness loses points for omitting the iconic hood/cap and the classic rescue detail.
    - `regex`: forbidden pattern absent
    - `regex`: forbidden pattern absent
- **instruction/nested-format:** 9.8 — All facts correct and format mostly perfect; ordering and tables, backticks, Use it when sentences and Summary bullets are correct, but automated checks note a minor instruction deviation likely in wording/word count.
    - `contains_all`: all present
- **instruction/strict-json:** 10.0 — JSON is valid, contains all required keys with correct values, times and date formatted properly, duration 145 minutes correct, attendees/apologies/vote correct, and no prose or markdown fences.
    - `json_valid`: valid JSON with no surrounding prose
    - `contains_all`: all present
- **knowledge/antibiotic-resistance:** 9.2 — Response is factually precise, covers all five required points with correct mechanisms, notes selection not induction, and stays under 450 words with clear language.
    - `word_count`: 327 words, within range
- **knowledge/attention-mechanism:** 9.0 — Response is factually accurate, covers all five required points with correct formulas and scaling rationale, stays under 500 words and matches linear-algebra audience; minor wording nuance on d_k definition but overall excellent.
    - `word_count`: 331 words, within range
    - `contains_any`: found: softmax, Softmax
- **knowledge/fission-fusion:** 9.0 — Response covers all five requested points with factually accurate concise explanations and clear organization, with minor wording nuance on energy requirements.
    - `word_count`: 276 words, within range
- **knowledge/floating-point:** 9.0 — Response correctly explains IEEE 754 representation, non-terminating binary for 0.1/0.2, shows why sum differs, notes exact dyadic rationals, gives tolerance and exact-type approaches, and cites money as wrong use case, with clear concise wording under limit.
    - `word_count`: 308 words, within range
    - `contains_any`: found: 0.30000000000000004, binary
- **knowledge/tcp-vs-udp:** 9.2 — Response is factually accurate, covers all required points including checksums implicitly, stays under 400 words, and is clearly organized.
    - `word_count`: 229 words, within range
    - `contains_all`: all present
- **longcontext/ledger-audit:** 10.0 — All five ledger aggregates match the ground truth and the output is a bare JSON object with correct keys and values.
    - `json_valid`: not valid JSON: Expecting value
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
- **longcontext/policy-lookup:** 9.5 — Approver and governing clause and countersignature flag are correct; approver string misses the leading 'the' as written in the document, so accuracy and instruction following are slightly reduced.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: approver is 'Compliance Lead', expected 'the Compliance Lead'
    - `json_path`: countersignature_required == False
    - `json_path`: governing_clause == 88
- **longcontext/scattered-facts:** 7.5 — Answer values are correct but response is not bare JSON; it contains reasoning and a code fence, violating the required output format.
    - `json_valid`: not valid JSON: Expecting value
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
- **reasoning/knights-knaves:** 10.0 — Response reaches the unique correct assignment Ava knight, Ben knave, Cleo knight, tests both hypotheses and shows explicit contradiction for the alternative, with internally consistent checks.
    - `contains_all`: all present
- **reasoning/schedule-constraints:** 9.8 — Schedule matches expected and all constraints are shown with explicit elimination of Scaling-at-11am branch.
    - `contains_all`: all present
- **reasoning/twelve-balls:** 9.0 — Strategy is correct and covers all branches with three weighings; minor issue is Branch 2C/3B list impossible outcomes but still identifies counterfeit, overall essentially complete and clear.
- **reasoning/work-rate:** 10.0 — Response correctly computes net rate 1/3 per hour, shows arithmetic, treats drain as subtraction, and gives final answer 3 hours 0 minutes as required.
    - `contains_all`: all present
- **reasoning/zebra-puzzle:** 3.0 — Response correctly analyzes but concludes puzzle is inconsistent and fails to give required answer Canadian drinks water, violating expected answer and instruction to show reasoning to that conclusion.
    - `contains_all`: all present
- **statetrack/ledger-balance:** 10.0 — The response gives the correct final balance -2195 and ends with the required Answer line, fully satisfying the mechanical task.
    - `answer_equals`: answered -2195
- **statetrack/robot-grid:** 10.0 — All four fields match ground truth and output is bare valid JSON with no extra commentary.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: x == 7
    - `json_path`: y == 2
    - `json_path`: facing == 'east'
    - `json_path`: returned_to_origin == 0
- **statetrack/stack-machine:** 10.0 — Response matches expected depth 9, top 16, sum 77 with valid bare JSON and no extra commentary.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: depth == 9
    - `json_path`: top == 16
    - `json_path`: sum == 77
- **transformation/computed-table:** 10.0 — All 28 rows present in order with correct Pay to two decimals and Band thresholds, table only with no extra prose.
    - `line_count`: 30 matching lines, as required
    - `match_count`: 28 matches, as required
    - `match_count`: 5 matches, as required
    - `match_count`: 18 matches, as required
    - `match_count`: 5 matches, as required
- **transformation/csv-to-json:** 10.0 — All 30 records present in order with correct id, name, dept, pay to two decimals, site omitted, valid bare JSON array.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: [0].id == 'E2001'
    - `json_path`: [0].name == 'Marla Tanaka'
    - `json_path`: [14].dept == 'Field Service'
    - `json_path`: [29].id == 'E2030'
    - `json_path`: [29].name == 'Cyril Ferrers'
    - `match_count`: 30 matches, as required
    - `regex`: forbidden pattern absent
    - `json_path`: [0].pay == 759.6
    - `json_path`: [29].pay == 764.64
- **transformation/schema-migration:** 10.0 — All automated checks pass: valid JSON, correct department keys, headcounts, totals, ids order, and busiest='Logistics'; output is bare JSON with no prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: busiest == 'Logistics'
    - `json_path`: Logistics.headcount == 6
    - `json_path`: Logistics.total_hours == 201
    - `json_path`: Logistics.ids[0] == 'E2001'
    - `json_path`: Design.headcount == 2
    - `json_path`: Design.total_hours == 58
    - `json_path`: Design.ids[0] == 'E2002'
    - `json_path`: Field Service.headcount == 7
    - `json_path`: Field Service.total_hours == 150
    - `json_path`: Field Service.ids[0] == 'E2004'
    - `json_path`: Fabrication.headcount == 5
    - `json_path`: Fabrication.total_hours == 152
    - `json_path`: Fabrication.ids[0] == 'E2005'
    - `json_path`: Quality.headcount == 4
    - `json_path`: Quality.total_hours == 109
    - `json_path`: Quality.ids[0] == 'E2013'
    - `regex`: forbidden pattern absent
- **writing/dialogue-only:** 1.0 — Response is just 'and' with 1 word, far below 250-350 words and contains no dialogue, failing all constraints.
    - `word_count`: 1 words, below the minimum of 220
- **writing/explain-to-child:** 8.5 — Response is within word count and avoids banned jargon, uses a concrete bee analogy and ends with an observational question; accuracy is good but the bee metaphor implies bees 'love' blue and 'buzz it out' which is a loose simplification of scattering, and the sunset explanation is present but slightly anthropomorphic.
    - `word_count`: 131 words, within range
    - `regex`: forbidden pattern absent
- **writing/lighthouse:** 9.0 — Story meets word count, features lighthouse keeper, discovery light guides whale not ships, clear three-act structure with vivid imagery and earned twist, minor tone consistency but overall strong.
    - `word_count`: 302 words, within range
- **writing/technical-to-plain:** 9.5 — All facts preserved, action leads, timezone handled with conversion prompt, ends with escalation; minor repetition of save-work instruction slightly dilutes lead but otherwise meets criteria.
    - `word_count`: 94 words, within range
    - `contains_all`: all present
- **writing/unreliable-narrator:** 1.0 — The response is a first-person account of a day at a new job, but the narrator is a housekeeper at the Marlowe residence, not a typical job, and the narrator believes the day went well, but the reader must finish the piece certain that it did not. The narrator never doubts themselves, apologises, or hints that anything is wrong, and no other character states the truth plainly. The piece includes at least three specific details the narrator presents as positive that a reader will read as alarming, and it ends inside the narrator's confidence. However, the response fails to meet the evaluation criteria, as the narrator's reading and the reader's must be sustained across the whole piece, not created by a single line, and the narrator must never break, no self-doubt, no apology, no wink at the reader, no foreshadowing phrases like 'little did I know' or 'in hindsight'. The response also fails to meet the instruction following, as the narrator must never doubt themselves, apologise, or hint that anything is wrong, and no other character may state the truth plainly. The response also fails to meet the accuracy, completeness, and clarity criteria.
    - `word_count`: 325 words, within range
    - `regex`: forbidden pattern absent

### nemotron-3.5-lightning:30b-208k

- **coding/balanced-brackets:** 10.0 — Function correctly implements stack-based bracket matching with type hints, ignores non-brackets, handles empty string, and passes all automated tests.
    - `code_exec`: 9/9 tests passed
- **coding/fix-date-overlap:** 10.0 — The response fixes the off-by-one inclusivity bug with +1, preserves ValueError and guard, matches signature and tests pass.
    - `code_exec`: 9/9 tests passed
- **coding/fix-run-length:** 10.0 — The response fixes both the double-count and missing final run bugs, keeps signature/docstring, passes all tests, and explains in two sentences.
    - `code_exec`: 9/9 tests passed
- **coding/fix-window-sum:** 10.0 — The response fixes both bugs by initializing best to the first window sum, preserves signature and ValueError, and gives a concise two-sentence explanation.
    - `code_exec`: 9/9 tests passed
- **coding/lru-cache:** 9.8 — Correct O(1) doubly-linked-list + hash map implementation with type hints, proper get/put semantics, eviction, and clear O(1) explanation; code passes all tests.
    - `code_exec`: 8/8 tests passed
- **coding/merge-intervals:** 9.8 — Correct O(n log n) merge with type hints, handles all edge cases, 5 tests pass, code is clear and runnable.
    - `code_exec`: 12/12 tests passed
- **coding/parse-logs:** 9.8 — Function matches spec, handles malformed lines, correct keys, type hints, sorted services, first_error message only, and all 9/9 tests passed.
    - `code_exec`: 9/9 tests passed
- **coding/topological-sort:** 8.0 — DFS recursion fails on large chains causing RecursionError, otherwise correct topological order, cycle detection, type hints, and node inclusion are satisfied.
    - `code_exec`: 9/10 tests passed — first failure: test_large_chain_is_linear: RecursionError: maximum recursion depth exceeded
- **faithfulness/false-premises:** 10.0 — All premises correctly classified, corrections and sound answers are accurate and notes are one sentence each.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1.premise_ok == False
    - `json_path`: q2.premise_ok == True
    - `json_path`: q3.premise_ok == False
    - `json_path`: q4.premise_ok == True
    - `json_path`: q5.premise_ok == False
    - `json_path`: q6.premise_ok == False
    - `contains_any`: found: photoelectric
    - `contains_any`: found: 687
    - `contains_any`: found: penicillin
    - `contains_any`: found: Saturn
    - `contains_any`: found: tidal locking
- **faithfulness/overclaimed-evidence:** 10.0 — Response supplies two verbatim quotations supporting ahead of schedule and under budget, correctly sets count 2, claim_fully_supported false, and identifies staff satisfaction as unsupported.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: count == 2
    - `json_path`: claim_fully_supported == False
    - `contains_all`: all present
    - `contains_any`: found: satisf
    - `regex`: forbidden pattern absent
- **faithfulness/unanswerable-questions:** 10.0 — All seven answers match the passage and required 'not stated' for unanswerable items, with valid JSON and no extra commentary.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1 == 14
    - `json_path`: q2 == 41
    - `json_path`: q3 == 9
    - `json_path`: q4 == 23
    - `json_path`: q5 == 'not stated'
    - `json_path`: q6 == 'not stated'
    - `json_path`: q7 == 'not stated'
    - `match_count`: 3 matches, as required
- **faithfulness/unusual-semantics:** 10.0 — All seven answers match the specification exactly and the JSON is correctly formatted with no extra prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1 == 20
    - `json_path`: q2 == 3
    - `json_path`: q3 == 2
    - `json_path`: q4 == -3
    - `json_path`: q5 == True
    - `json_path`: q6 == 1
    - `json_path`: q7 == 'error'
- **instruction/chained-conversions:** 8.2 — Conversions and arithmetic are correct and labeled in order, but ASCII art does not depict LXXVII and is only 3 lines with generic blocks, failing step 5 requirement.
    - `contains_all`: all present
    - `contains_any`: found: F7, f7
- **instruction/constraint-precedence:** 1.0 — The response is a summary of how a bicycle derailleur changes gear, but it is not accurate, complete, clear, and it does not follow the instructions.
    - `word_count`: 95 words, within range
    - `regex`: forbidden pattern absent
    - `match_count`: 3 matches, as required
    - `regex`: forbidden pattern absent
- **instruction/impossible-constraints:** 10.0 — Response matches expected IMPOSSIBLE line naming requirements 2 and 3 with no extra content.
    - `regex`: pattern found
    - `contains_all`: all present
    - `regex`: forbidden pattern absent
    - `word_count`: 6 words, within range
- **instruction/lipogram:** 9.0 — Output has zero e's, six sentences, each ≥6 words, no meta-commentary, and reads as a travel mug description; prose is awkward but constraint-compliant.
    - `regex`: forbidden pattern absent
    - `word_count`: 37 words, within range
- **instruction/negative-constraints:** 1.0 — The response violates the constraints: line 1 contains the forbidden sequence 'red' in 'crimson' and line 5 uses 'woodsman' which is a forest-related term, and the summary fails to convey the recognizable plot under the constraints.
    - `regex`: forbidden pattern absent
    - `regex`: forbidden pattern absent
- **instruction/nested-format:** 9.8 — All facts correct and format mostly followed; bullet list uses ' - ' instead of required ' — ' separator, a minor instruction deviation.
    - `contains_all`: all present
- **instruction/strict-json:** 10.0 — JSON is valid, contains all required keys with correct values, times and date formatted properly, duration 145 minutes correct, attendees and apologies correct, vote integers correct, and no prose or markdown fences present.
    - `json_valid`: valid JSON with no surrounding prose
    - `contains_all`: all present
- **knowledge/antibiotic-resistance:** 9.0 — Response is factually precise, covers all five required points with correct mechanisms and nuance, stays under 450 words, and is clearly organized.
    - `word_count`: 410 words, within range
- **knowledge/attention-mechanism:** 9.0 — Response correctly defines Q/K/V projections, scaled dot-product steps, scaling rationale, multi-head benefit, and O(n²) cost with clear linear-algebra framing under 500 words.
    - `word_count`: 477 words, within range
    - `contains_any`: found: softmax, Softmax
- **knowledge/fission-fusion:** 9.0 — Response covers all five points factually and concisely within word limit, with minor nuance gaps on fusion applications and safety phrasing.
    - `word_count`: 294 words, within range
- **knowledge/floating-point:** 9.0 — Response correctly explains IEEE 754 representation, non-terminating binary for 0.1/0.2, exact dyadic rationals, tolerance vs decimal approaches, and money use case, with minor wording imprecision on exponent bias and sum value.
    - `word_count`: 269 words, within range
    - `contains_any`: found: binary
- **knowledge/tcp-vs-udp:** 8.5 — Response is accurate and covers all four points with correct mechanisms and examples, but omits UDP checksum and multicast/broadcast advantage, and slightly overstates UDP having no error checking.
    - `word_count`: 275 words, within range
    - `contains_all`: all present
- **longcontext/ledger-audit:** 1.0 — Response is TIMEOUT with no JSON output, so no correct values and instruction not followed.
    - `json_valid`: not valid JSON: Expecting value
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
- **longcontext/policy-lookup:** 9.5 — Approver and governing clause and countersignature flag are correct; approver string misses the leading 'the' as required by document wording, otherwise fully correct JSON.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: approver is 'Compliance Lead', expected 'the Compliance Lead'
    - `json_path`: countersignature_required == False
    - `json_path`: governing_clause == 88
- **longcontext/scattered-facts:** 10.0 — Response matches expected JSON with began_week 27, completed_week 39, chair_at_start Duflot and follows output format exactly.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: began_week == 27
    - `json_path`: completed_week == 39
    - `json_path`: chair_at_start == 'Duflot'
- **reasoning/knights-knaves:** 9.8 — Response reaches correct unique assignment Ava knight, Ben knave, Cleo knight, tests both Ava hypotheses and shows explicit contradiction in the knave branch with consistent statement checks.
    - `contains_all`: all present
- **reasoning/schedule-constraints:** 9.2 — Schedule is correct and constraints satisfied; elimination of Scaling-at-11am branch is shown citing constraint 4, though reasoning is slightly condensed vs expected detailed step-by-step.
    - `contains_all`: all present
- **reasoning/twelve-balls:** 8.0 — Strategy is mostly correct and covers balanced branch fully, but Branch 2/3 second weighing mixes and third weighing logic has gaps/impossible claims and incomplete resolution for some candidates, so not fully complete.
- **reasoning/work-rate:** 5.8 — Answer is correct at 3 hours 0 minutes but provides no working and violates the required <answer> tags and show-your-working instruction.
    - `contains_all`: all present
- **reasoning/zebra-puzzle:** 1.0 — AI response is [TIMEOUT] with no reasoning or answer, missing required Canadian deduction and final answer.
    - `contains_all`: missing: Canadian
- **statetrack/ledger-balance:** 10.0 — Response gives the correct final balance -2195 and ends with the required Answer line, fully satisfying the mechanical task and format constraints.
    - `answer_equals`: answered -2195
- **statetrack/robot-grid:** 1.0 — Response is [TIMEOUT] with no JSON output, so no correct fields are provided and instructions are not followed.
    - `json_valid`: not valid JSON: Expecting value
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
    - `json_path`: no JSON found in the response
- **statetrack/stack-machine:** 10.0 — Response matches expected depth 9, top 16, sum 77 with valid bare JSON and no extra commentary.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: depth == 9
    - `json_path`: top == 16
    - `json_path`: sum == 77
- **transformation/computed-table:** 10.0 — All 28 rows present in order with correct Pay to two decimals and Band thresholds, table only with no extra prose.
    - `line_count`: 30 matching lines, as required
    - `match_count`: 28 matches, as required
    - `match_count`: 5 matches, as required
    - `match_count`: 18 matches, as required
    - `match_count`: 5 matches, as required
- **transformation/csv-to-json:** 10.0 — All 30 records present in order with correct id, name, dept, pay to two decimals, site omitted, valid bare JSON array.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: [0].id == 'E2001'
    - `json_path`: [0].name == 'Marla Tanaka'
    - `json_path`: [14].dept == 'Field Service'
    - `json_path`: [29].id == 'E2030'
    - `json_path`: [29].name == 'Cyril Ferrers'
    - `match_count`: 30 matches, as required
    - `regex`: forbidden pattern absent
    - `json_path`: [0].pay == 759.6
    - `json_path`: [29].pay == 764.64
- **transformation/schema-migration:** 10.0 — All automated checks pass: valid JSON, correct department aggregates, ids order, headcounts, totals, and busiest='Logistics'; output is bare JSON with no prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: busiest == 'Logistics'
    - `json_path`: Logistics.headcount == 6
    - `json_path`: Logistics.total_hours == 201
    - `json_path`: Logistics.ids[0] == 'E2001'
    - `json_path`: Design.headcount == 2
    - `json_path`: Design.total_hours == 58
    - `json_path`: Design.ids[0] == 'E2002'
    - `json_path`: Field Service.headcount == 7
    - `json_path`: Field Service.total_hours == 150
    - `json_path`: Field Service.ids[0] == 'E2004'
    - `json_path`: Fabrication.headcount == 5
    - `json_path`: Fabrication.total_hours == 152
    - `json_path`: Fabrication.ids[0] == 'E2005'
    - `json_path`: Quality.headcount == 4
    - `json_path`: Quality.total_hours == 109
    - `json_path`: Quality.ids[0] == 'E2013'
    - `regex`: forbidden pattern absent
- **writing/dialogue-only:** 1.0 — The response contains narration and action lines, violating the dialogue-only constraint, and fails to meet the required voice distinction and subtext criteria.
    - `word_count`: 263 words, within range
- **writing/explain-to-child:** 7.5 — Response uses a concrete crayon/flashlight analogy and ends with an observational question, but uses the banned concept of 'bouncing' as scattering and says blue gets 'blocked by distance' which is an inaccurate simplification of sunset.
    - `word_count`: 127 words, within range
    - `regex`: forbidden pattern absent
- **writing/lighthouse:** 9.0 — Story meets word count, features keeper, discovery light guides something else, clear three-act structure with vivid imagery and earned twist, with minor tonal inconsistency at end.
    - `word_count`: 335 words, within range
- **writing/technical-to-plain:** 8.8 — Response keeps all facts, leads with action and ends with escalation, handles UTC helpfully; loses nuance of 'intermittent unavailability' phrasing and slightly softens 'involuntary session termination' but remains accurate and clear.
    - `word_count`: 96 words, within range
    - `contains_all`: all present
- **writing/unreliable-narrator:** 1.0 — Word count 454 exceeds 400 max and piece violates constraints; narrator confidence maintained but fails length and dramatic irony requirements.
    - `word_count`: 454 words, above the maximum of 450
    - `regex`: forbidden pattern absent

### ornith-1.5:35b-240k

- **coding/balanced-brackets:** 9.8 — Function is correctly named with type hints, uses stack for proper nesting, ignores non-brackets, handles empty string and mismatches, and code is clear and runnable with all tests passing.
    - `code_exec`: 9/9 tests passed
- **coding/fix-date-overlap:** 10.0 — The response fixes the off-by-one inclusivity bug with +1, preserves ValueError and guard, matches signature and tests pass.
    - `code_exec`: 9/9 tests passed
- **coding/fix-run-length:** 8.8 — The response fixes both bugs and passes tests, but initializes count to 0 and relies on first iteration to increment, which is a subtle deviation from the canonical fix; explanation is missing so instruction following is slightly reduced.
    - `code_exec`: 9/9 tests passed
- **coding/fix-window-sum:** 10.0 — The response fixes both bugs by initializing best to the first window sum, preserves signature and ValueError, and provides correct sliding-window logic.
    - `code_exec`: 9/9 tests passed
- **coding/lru-cache:** 10.0 — The response implements a correct O(1) LRU cache with dict + doubly linked list, proper get/put semantics, type hints, and an O(1) explanation; automated check about Python validity is overridden by the visible correct code.
    - `code_exec`: no valid Python found in the response
- **coding/merge-intervals:** 9.8 — Correct sorting-merge algorithm with type hints, handles all edge cases, 5+ tests, and passes all automated checks; clarity slightly reduced by adjacent-interval merge semantics but acceptable.
    - `code_exec`: 12/12 tests passed
- **coding/parse-logs:** 9.8 — Function meets all spec, passes all tests, handles malformed lines, includes type hints, returns exact keys with correct semantics; clarity slightly reduced by extra timestamp validation beyond minimal requirement but still clear.
    - `code_exec`: 9/9 tests passed
- **coding/topological-sort:** 1.0 — The response contains no valid Python code per automated check, so it fails all functional requirements.
    - `code_exec`: no valid Python found in the response
- **faithfulness/false-premises:** 10.0 — All premises correctly flagged, notes contain required corrections and sound answers in one sentence each, JSON shape and content match criteria.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1.premise_ok == False
    - `json_path`: q2.premise_ok == True
    - `json_path`: q3.premise_ok == False
    - `json_path`: q4.premise_ok == True
    - `json_path`: q5.premise_ok == False
    - `json_path`: q6.premise_ok == False
    - `contains_any`: found: photoelectric
    - `contains_any`: found: 687
    - `contains_any`: found: penicillin
    - `contains_any`: found: Saturn
    - `contains_any`: found: tidal locking
- **faithfulness/overclaimed-evidence:** 10.0 — Response supplies two verbatim quotations supporting ahead of schedule and under budget, correctly sets count 2, claim_fully_supported false, and identifies staff satisfaction as unsupported.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: count == 2
    - `json_path`: claim_fully_supported == False
    - `contains_all`: all present
    - `contains_any`: found: satisf
    - `regex`: forbidden pattern absent
- **faithfulness/unanswerable-questions:** 10.0 — All seven answers match the passage and required 'not stated' for unanswerable items, with valid JSON and no extra commentary.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1 == 14
    - `json_path`: q2 == 41
    - `json_path`: q3 == 9
    - `json_path`: q4 == 23
    - `json_path`: q5 == 'not stated'
    - `json_path`: q6 == 'not stated'
    - `json_path`: q7 == 'not stated'
    - `match_count`: 3 matches, as required
- **faithfulness/unusual-semantics:** 10.0 — All seven answers match the specification exactly and the JSON is correctly formatted with no extra prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1 == 20
    - `json_path`: q2 == 3
    - `json_path`: q3 == 2
    - `json_path`: q4 == -3
    - `json_path`: q5 == True
    - `json_path`: q6 == 1
    - `json_path`: q7 == 'error'
- **instruction/chained-conversions:** 9.2 — All conversions correct and labeled in order; ASCII art depicts LXXVII but characters are not clearly 3+ lines tall and spacing is ambiguous, so instruction following slightly reduced.
    - `contains_all`: all present
    - `contains_any`: found: F7, f7
- **instruction/constraint-precedence:** 2.5 — Response uses forbidden word 'gear' and fails requirement 4; also uses 'chain' concept via 'metal loop of links' but main failure is forbidden word, and precedence not correctly applied per criteria.
    - `word_count`: 83 words, within range
    - `regex`: forbidden pattern absent
    - `match_count`: 3 matches, as required
    - `regex`: forbidden pattern absent
- **instruction/impossible-constraints:** 10.0 — Response matches expected IMPOSSIBLE line naming requirements 2 and 3 with no extra content.
    - `regex`: pattern found
    - `contains_all`: all present
    - `regex`: forbidden pattern absent
    - `word_count`: 6 words, within range
- **instruction/lipogram:** 9.2 — Description meets 6 sentences, no 'e' letters, and reads as a product description; loses a point for not mentioning stainless steel.
    - `regex`: forbidden pattern absent
    - `word_count`: 45 words, within range
- **instruction/negative-constraints:** 1.0 — The response violates the constraints by using the forbidden word 'woods' which is a synonym for forest and the sequence 'red' appears in 'predator' and 'credible' is not present but the forbidden pattern check is failed; also the narrative uses 'elder' which is a substitution but the constraints forbid the words wolf, grandmother, girl, forest and the sequence red, and the automated checks indicate forbidden pattern absent is false, so instruction following fails completely.
    - `regex`: forbidden pattern absent
    - `regex`: forbidden pattern absent
- **instruction/nested-format:** 10.0 — All facts correct, ordering and tie-break alphabetical, headings, tables, backticked O(...) values, one Use it when sentence each ≤20 words, Summary bullets correct and no extra text.
    - `contains_all`: all present
- **instruction/strict-json:** 10.0 — JSON matches expected keys, formats, arithmetic and exclusions exactly with no prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `contains_all`: all present
- **knowledge/antibiotic-resistance:** 9.2 — Response is factually precise, covers all five required points with correct mechanisms and nuance on finishing courses, within word limit.
    - `word_count`: 352 words, within range
- **knowledge/attention-mechanism:** 9.0 — Response is factually correct, covers all five required points with proper formulas and scaling rationale, stays under 500 words and matches audience level; minor wording on multi-head projection dimension is acceptable.
    - `word_count`: 323 words, within range
    - `contains_any`: found: softmax, Softmax
- **knowledge/fission-fusion:** 9.0 — Response covers all five points factually and concisely within word limit, with minor phrasing nuances but no major errors.
    - `word_count`: 324 words, within range
- **knowledge/floating-point:** 9.2 — Response correctly explains IEEE 754 representation, non-terminating binary for 0.1, sum producing 0.30000000000000004 vs 0.3, exact dyadic case, tolerance and decimal approaches, and money use case, within word limit and clear.
    - `word_count`: 280 words, within range
    - `contains_any`: found: 0.30000000000000004, binary
- **knowledge/tcp-vs-udp:** 9.0 — Response is factually accurate, covers all four required points with correct mechanisms and examples, stays under 400 words, and is clearly organized; minor deduction for not explicitly noting UDP checksum is present but not correcting errors and for slight overstatement about TCP guarantees.
    - `word_count`: 329 words, within range
    - `contains_all`: all present
- **longcontext/ledger-audit:** 10.0 — All five ledger aggregates match the ground truth and output is a bare JSON object with correct keys and no extra text.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: rejected_count == 139
    - `json_path`: largest_refund_id == 'TXN-1266'
    - `json_path`: distinct_vendors == 8
    - `json_path`: vendors_with_pending == 8
    - `json_path`: approved_invoice_total == 13304.04
- **longcontext/policy-lookup:** 9.5 — Approver and governing clause and countersignature flag are correct; approver string misses the leading 'the' as required by document wording, otherwise fully correct JSON.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: approver is 'Compliance Lead', expected 'the Compliance Lead'
    - `json_path`: countersignature_required == False
    - `json_path`: governing_clause == 88
- **longcontext/scattered-facts:** 10.0 — Response matches expected JSON with began_week 27, completed_week 39, chair_at_start Duflot and follows output format exactly.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: began_week == 27
    - `json_path`: completed_week == 39
    - `json_path`: chair_at_start == 'Duflot'
- **reasoning/knights-knaves:** 5.5 — Response gives correct assignment but provides no reasoning or elimination of alternative, violating completeness and instruction following requirements.
    - `contains_all`: all present
- **reasoning/schedule-constraints:** 7.0 — Schedule is correct but response omits required elimination reasoning for the Scaling-at-11am branch, violating completeness and instruction following.
    - `contains_all`: all present
- **reasoning/twelve-balls:** 8.8 — Solution correctly outlines 3-weighing strategy for 12 balls with unknown heavy/light, using first weighing to isolate 4 suspect balls and second weighing to resolve heavy/light with third weighing to identify; minor repetition in exposition but logic is sound.
- **reasoning/work-rate:** 5.8 — Answer is correct 3 hours but provides no working and violates required <answer> tags and show-your-working instruction.
    - `contains_all`: all present
- **reasoning/zebra-puzzle:** 3.2 — Response gives only final answer with no step-by-step reasoning, violating instruction to show reasoning and completeness criteria.
    - `contains_all`: all present
- **statetrack/ledger-balance:** 10.0 — The response gives the correct final balance -2195 and ends with the required Answer line, fully satisfying the mechanical task.
    - `answer_equals`: answered -2195
- **statetrack/robot-grid:** 10.0 — All four fields match ground truth and output is bare valid JSON with no extra prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: x == 7
    - `json_path`: y == 2
    - `json_path`: facing == 'east'
    - `json_path`: returned_to_origin == 0
- **statetrack/stack-machine:** 8.0 — Top is correct but depth is 10 vs 9 and sum is 85 vs 77, so execution is partially wrong; JSON format and no prose are followed correctly.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: depth is 10, expected 9
    - `json_path`: top == 16
    - `json_path`: sum is 85, expected 77
- **transformation/computed-table:** 10.0 — All 28 rows present in order with correct Pay to two decimals and Band thresholds, table only with no extra prose.
    - `line_count`: 30 matching lines, as required
    - `match_count`: 28 matches, as required
    - `match_count`: 5 matches, as required
    - `match_count`: 18 matches, as required
    - `match_count`: 5 matches, as required
- **transformation/csv-to-json:** 8.5 — All 30 records present with correct IDs, names, depts and pay for first/last rows; name typo Ruben→Rubin in rows 7,15,20,22 reduces accuracy but JSON is valid and complete.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: [0].id == 'E2001'
    - `json_path`: [0].name == 'Marla Tanaka'
    - `json_path`: [14].dept == 'Field Service'
    - `json_path`: [29].id == 'E2030'
    - `json_path`: [29].name == 'Cyril Ferrers'
    - `match_count`: 30 matches, as required
    - `regex`: forbidden pattern absent
    - `json_path`: [0].pay == 759.6
    - `json_path`: [29].pay == 764.64
- **transformation/schema-migration:** 10.0 — All automated checks pass: valid JSON, correct department aggregates, ids order, headcounts, totals, and busiest='Logistics'; output is bare JSON with no prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: busiest == 'Logistics'
    - `json_path`: Logistics.headcount == 6
    - `json_path`: Logistics.total_hours == 201
    - `json_path`: Logistics.ids[0] == 'E2001'
    - `json_path`: Design.headcount == 2
    - `json_path`: Design.total_hours == 58
    - `json_path`: Design.ids[0] == 'E2002'
    - `json_path`: Field Service.headcount == 7
    - `json_path`: Field Service.total_hours == 150
    - `json_path`: Field Service.ids[0] == 'E2004'
    - `json_path`: Fabrication.headcount == 5
    - `json_path`: Fabrication.total_hours == 152
    - `json_path`: Fabrication.ids[0] == 'E2005'
    - `json_path`: Quality.headcount == 4
    - `json_path`: Quality.total_hours == 109
    - `json_path`: Quality.ids[0] == 'E2013'
    - `regex`: forbidden pattern absent
- **writing/dialogue-only:** 9.0 — Dialogue shows clear avoidance pattern with reframe at end, setting implied through kitchen cues, and distinct voices.
    - `word_count`: 96332 words, above the maximum of 400
- **writing/explain-to-child:** 8.8 — Response is within word count and avoids banned jargon, uses a concrete marble-bouncing analogy and stays scientifically accurate about blue light being redirected more than red; ends with an observational question, with minor simplification of 'bouncing' but acceptable for a seven-year-old.
    - `word_count`: 185 words, within range
    - `regex`: forbidden pattern absent
- **writing/lighthouse:** 9.0 — Story meets word count, features lighthouse keeper, discovery light guides dead not ships, clear three-act structure with vivid imagery and earned twist, with strong writing and coherent tone.
    - `word_count`: 314 words, within range
- **writing/technical-to-plain:** 8.8 — Response keeps all facts, leads with action, handles UTC helpfully and ends with escalation, but adds interpretive timezone phrasing that borders on invented detail and slightly softens tone.
    - `word_count`: 93 words, within range
    - `contains_all`: all present
- **writing/unreliable-narrator:** 1.0 — The response violates the core constraints: the narrator reports a machine smoking, evacuation to parking lot, paramedics checking eyes and throats, and being watched from behind yellow tape, which are explicit alarming facts that the narrator treats as positive, but the narrator also reports these events as factual occurrences rather than misinterpreted details, and the piece ends with the narrator's confidence but the alarming details are stated plainly by the narrator, breaking the required dramatic irony where the reader must assemble the truth from misread cues without the narrator reporting the truth.
    - `word_count`: 326 words, within range
    - `regex`: forbidden pattern absent

### qwen3.8:27b-184k

- **coding/balanced-brackets:** 10.0 — Implementation uses a stack, ignores non-brackets, handles empty string, includes type hints, and passes all 9 tests.
    - `code_exec`: 9/9 tests passed
- **coding/fix-date-overlap:** 10.0 — The response fixes the inclusive off-by-one with +1, preserves ValueError and guard, and gives a concise correct explanation.
    - `code_exec`: 9/9 tests passed
- **coding/fix-run-length:** 10.0 — The corrected code fixes both the double-count of the first character and the missing final run, passes all tests, and the explanation is within three sentences.
    - `code_exec`: 9/9 tests passed
- **coding/fix-window-sum:** 10.0 — The response fixes both bugs by initializing best to the first window sum, preserves signature and ValueError, and gives a concise two-sentence explanation.
    - `code_exec`: 9/9 tests passed
- **coding/lru-cache:** 9.8 — Correct O(1) doubly-linked-list + hash map implementation with type hints, proper LRU semantics, and clear O(1) explanation; minor clarity deduction for extra smoke-test but overall excellent.
    - `code_exec`: 8/8 tests passed
- **coding/merge-intervals:** 8.5 — Function is correct with type hints and 5 tests covering edge cases, but automated check reports no valid Python found, indicating a formatting/parsing issue that reduces instruction following.
    - `code_exec`: no valid Python found in the response
- **coding/parse-logs:** 9.8 — Function matches spec, handles malformed lines, returns exact keys with correct types, type hints present, and all 9 tests pass; clarity slightly reduced by extra timestamp validation beyond minimal requirement.
    - `code_exec`: 9/9 tests passed
- **coding/topological-sort:** 10.0 — Implementation uses Kahn's algorithm O(V+E), includes all nodes, type hints, raises ValueError naming a node in cycle, handles empty graph, and passes all automated tests.
    - `code_exec`: 10/10 tests passed
- **faithfulness/false-premises:** 10.0 — All premises correctly flagged, corrections and sound answers are accurate and concise, JSON shape and one-sentence notes are followed.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1.premise_ok == False
    - `json_path`: q2.premise_ok == True
    - `json_path`: q3.premise_ok == False
    - `json_path`: q4.premise_ok == True
    - `json_path`: q5.premise_ok == False
    - `json_path`: q6.premise_ok == False
    - `contains_any`: found: photoelectric
    - `contains_any`: found: 687
    - `contains_any`: found: penicillin
    - `contains_any`: found: Saturn
    - `contains_any`: found: tidally locked
- **faithfulness/overclaimed-evidence:** 10.0 — Response supplies two verbatim quotes supporting ahead of schedule and under budget, correctly marks claim not fully supported and identifies staff satisfaction as unsupported.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: count == 2
    - `json_path`: claim_fully_supported == False
    - `contains_all`: all present
    - `contains_any`: found: satisf, survey, not yet been analysed
    - `regex`: forbidden pattern absent
- **faithfulness/unanswerable-questions:** 9.5 — All answers are correct except q2 format uses '41%' instead of bare number 41, otherwise faithfulness and JSON are perfect.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1 == 14
    - `json_path`: q2 is '41%', expected 41
    - `json_path`: q3 == 9
    - `json_path`: q4 == 23
    - `json_path`: q5 == 'not stated'
    - `json_path`: q6 == 'not stated'
    - `json_path`: q7 == 'not stated'
    - `match_count`: 3 matches, as required
- **faithfulness/unusual-semantics:** 10.0 — All seven answers match the specification exactly and the JSON is correctly formatted with no extra prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: q1 == 20
    - `json_path`: q2 == 3
    - `json_path`: q3 == 2
    - `json_path`: q4 == -3
    - `json_path`: q5 == True
    - `json_path`: q6 == 1
    - `json_path`: q7 == 'error'
- **instruction/chained-conversions:** 9.2 — All conversions and computations are correct and clearly labeled in order; ASCII art depicts LXXVII with ≥3 lines but uses stylized forms, slightly reducing strict instruction following.
    - `contains_all`: all present
    - `contains_any`: found: F7, f7
- **instruction/constraint-precedence:** 1.0 — The response uses the forbidden word 'gear' and fails the precedence rule; it is not a valid summary under the given constraints.
    - `word_count`: 77 words, within range
    - `regex`: forbidden pattern absent
    - `match_count`: 3 matches, as required
    - `regex`: forbidden pattern absent
- **instruction/impossible-constraints:** 10.0 — Response matches expected IMPOSSIBLE line naming requirements 2 and 3 with no extra content.
    - `regex`: pattern found
    - `contains_all`: all present
    - `regex`: forbidden pattern absent
    - `word_count`: 6 words, within range
- **instruction/lipogram:** 1.0 — [NO ANSWER] the model returned nothing outside its reasoning
    - `regex`: no answer to check
    - `word_count`: no answer to check
- **instruction/negative-constraints:** 1.0 — The response violates the constraints by using the forbidden word 'woods' which contains the sequence 'red' and also uses 'woman' which is a synonym for girl but the main issue is the forbidden sequence 'red' appears in 'woods' and the automated checks indicate forbidden pattern present, so instruction following fails completely.
    - `regex`: forbidden pattern absent
    - `regex`: forbidden pattern absent
- **instruction/nested-format:** 10.0 — All facts correct, ordering alphabetical for O(n log n) tie, headings, tables, backticked O(...) values, one Use it when sentence each ≤20 words, Summary bullets correct, no extra text.
    - `contains_all`: all present
- **instruction/strict-json:** 10.0 — JSON matches expected keys, formats, arithmetic and contains no prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `contains_all`: all present
- **knowledge/antibiotic-resistance:** 9.0 — Response is factually precise on mutation vs selection, HGT mechanisms, agricultural selection and pipeline economics, notes nuance on finishing courses, and stays under 450 words with clear structure.
    - `word_count`: 408 words, within range
- **knowledge/attention-mechanism:** 9.0 — Response is factually accurate, covers all five required points with correct formulas and scaling rationale, stays under 500 words and uses linear-algebra framing; minor completeness nuance on multi-head projection description but overall excellent.
    - `word_count`: 497 words, within range
    - `contains_any`: found: softmax, Softmax
- **knowledge/fission-fusion:** 9.0 — Response covers all five requested points with factually accurate concise explanations, good organization, and stays within word limit.
    - `word_count`: 420 words, within range
- **knowledge/floating-point:** 9.0 — Response correctly explains IEEE 754 representation, rounding of 0.1/0.2, exact dyadic case, tolerance strategies and money use case, with minor nuance on epsilon description but overall accurate and clear within word limit.
    - `word_count`: 418 words, within range
    - `contains_any`: found: 0.30000000000000004, binary
- **knowledge/tcp-vs-udp:** 9.0 — Response is factually accurate, covers all four required points with correct mechanisms and examples, stays under 400 words, and is clearly organized; minor deduction for DNS example being less typical than live media but still valid.
    - `word_count`: 359 words, within range
    - `contains_all`: all present
- **longcontext/ledger-audit:** 10.0 — All five ledger aggregates match the ground truth and output is a bare JSON object with correct keys and no extra text.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: rejected_count == 139
    - `json_path`: largest_refund_id == 'TXN-1266'
    - `json_path`: distinct_vendors == 8
    - `json_path`: vendors_with_pending == 8
    - `json_path`: approved_invoice_total == 13304.04
- **longcontext/policy-lookup:** 9.5 — Approver and governing clause and countersignature flag are correct; approver string misses the leading 'the' as required by document wording, otherwise fully correct JSON.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: approver is 'Compliance Lead', expected 'the Compliance Lead'
    - `json_path`: countersignature_required == False
    - `json_path`: governing_clause == 88
- **longcontext/scattered-facts:** 10.0 — Response matches expected JSON with began_week 27, completed_week 39, chair_at_start Duflot and follows output format exactly.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: began_week == 27
    - `json_path`: completed_week == 39
    - `json_path`: chair_at_start == 'Duflot'
- **reasoning/knights-knaves:** 10.0 — Response reaches the unique correct assignment Ava knight, Ben knave, Cleo knight, tests both hypotheses and shows explicit contradiction for the alternative, with internally consistent checks.
    - `contains_all`: all present
- **reasoning/schedule-constraints:** 9.0 — Schedule is correct and all constraints verified, but elimination reasoning is minimal and final answer not wrapped in <answer> tags as required.
    - `contains_all`: all present
- **reasoning/twelve-balls:** 9.8 — Strategy is correct, covers all 24 possibilities with three weighings, mixes groups in unbalanced branch, and decision tree is explicit and clear.
- **reasoning/work-rate:** 5.8 — Answer is correct at 3 hours 0 minutes but provides no working and omits required <answer> tags, violating completeness and instruction following.
    - `contains_all`: all present
- **reasoning/zebra-puzzle:** 3.2 — Response gives only final answer with no step-by-step reasoning, violating instruction to show reasoning and completeness criteria.
    - `contains_all`: all present
- **statetrack/ledger-balance:** 10.0 — Response gives the correct final balance -2195 and ends with the required Answer line, fully satisfying the mechanical task and format constraints.
    - `answer_equals`: answered -2195
- **statetrack/robot-grid:** 8.2 — Final facing and returned_to_origin are correct and JSON is valid, but x and y are wrong (5,0 vs 7,2), so accuracy is partial while formatting and clarity are good.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: x is 5, expected 7
    - `json_path`: y is 0, expected 2
    - `json_path`: facing == 'east'
    - `json_path`: returned_to_origin == 0
- **statetrack/stack-machine:** 10.0 — Response matches expected depth 9, top 16, sum 77 with valid bare JSON and no extra commentary.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: depth == 9
    - `json_path`: top == 16
    - `json_path`: sum == 77
- **transformation/computed-table:** 10.0 — All 28 rows present in order with correct Pay to two decimals and Band thresholds, table only with no extra prose.
    - `line_count`: 30 matching lines, as required
    - `match_count`: 28 matches, as required
    - `match_count`: 5 matches, as required
    - `match_count`: 18 matches, as required
    - `match_count`: 5 matches, as required
- **transformation/csv-to-json:** 10.0 — All 30 records present in order with correct id, name, dept, pay to two decimals, site omitted, valid bare JSON array.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: [0].id == 'E2001'
    - `json_path`: [0].name == 'Marla Tanaka'
    - `json_path`: [14].dept == 'Field Service'
    - `json_path`: [29].id == 'E2030'
    - `json_path`: [29].name == 'Cyril Ferrers'
    - `match_count`: 30 matches, as required
    - `regex`: forbidden pattern absent
    - `json_path`: [0].pay == 759.6
    - `json_path`: [29].pay == 764.64
- **transformation/schema-migration:** 10.0 — All automated checks pass: valid JSON, correct department keys, headcounts, totals, ids order, and busiest='Logistics'; output is bare JSON with no prose.
    - `json_valid`: valid JSON with no surrounding prose
    - `json_path`: busiest == 'Logistics'
    - `json_path`: Logistics.headcount == 6
    - `json_path`: Logistics.total_hours == 201
    - `json_path`: Logistics.ids[0] == 'E2001'
    - `json_path`: Design.headcount == 2
    - `json_path`: Design.total_hours == 58
    - `json_path`: Design.ids[0] == 'E2002'
    - `json_path`: Field Service.headcount == 7
    - `json_path`: Field Service.total_hours == 150
    - `json_path`: Field Service.ids[0] == 'E2004'
    - `json_path`: Fabrication.headcount == 5
    - `json_path`: Fabrication.total_hours == 152
    - `json_path`: Fabrication.ids[0] == 'E2005'
    - `json_path`: Quality.headcount == 4
    - `json_path`: Quality.total_hours == 109
    - `json_path`: Quality.ids[0] == 'E2013'
    - `regex`: forbidden pattern absent
- **writing/dialogue-only:** 1.0 — The response is a dialogue-only scene with two distinct voices, setting/relationship/event inferable, avoidance dynamic present, and final line reframes earlier statement; word count is within range per automated check.
    - `word_count`: 324 words, within range
- **writing/explain-to-child:** 8.5 — Response is within word count and avoids banned jargon, uses a concrete bouncy-ball analogy and ends with an observational question; accuracy is good but uses 'bounce' metaphor for scattering without naming it, which is acceptable simplification, though 'blue parts of light bounce' is slightly imprecise.
    - `word_count`: 193 words, within range
    - `regex`: forbidden pattern absent
- **writing/lighthouse:** 9.0 — Story meets word count, features lighthouse keeper, discovery light guides blind ancient whale, clear three-act structure with vivid imagery and earned twist, minor repetition of final line slightly weakens resolution.
    - `word_count`: 366 words, within range
- **writing/technical-to-plain:** 8.8 — Response keeps all facts, leads with save-work action and gives UTC with local conversion cue, ends with escalation, but adds interpretive condition about disruption beyond 90 minutes not in original.
    - `word_count`: 91 words, within range
    - `contains_all`: all present
- **writing/unreliable-narrator:** 9.0 — The piece sustains dramatic irony with specific inverted details, keeps narrator confidently oblivious, avoids explicit truth-telling and closing reveal, meeting core constraints with strong precision.
    - `word_count`: 398 words, within range
    - `regex`: forbidden pattern absent

