# instruction / nested-format

**Prompt:** Produce a reference card for four sorting algorithms: insertion sort, merge sort, quicksort, and heapsort.

Format requirements — follow all of them exactly:

1. Order the algorithms from best to worst average-case time complexity. Break ties alphabetically.
2. For each algorithm, output a level-3 markdown heading containing only the algorithm's name in Title Case.
3. Under each heading, a markdown table with exactly two columns headed `Property` and `Value`, and exactly four rows in this order: Best, Average, Worst, Space.
4. Every complexity value must be written in the form `O(...)` inside backticks.
5. After each table, exactly one sentence beginning with the words "Use it when" — no more than 20 words.
6. After all four sections, add a final level-3 heading `Summary` followed by exactly one bullet list of four items, each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`.

Output nothing else. No introduction, no conclusion.

## Repeat 1 (seed 0)

- Score: 9.75/10 — All facts correct and format mostly perfect; ordering and tables, backticks, Use it when sentences and Summary bullets are correct, but automated checks note a minor instruction deviation likely in wording/word count.
- Tokens/s: 56.2
- Prefill tok/s: 1273.0
- TTFT: 0.33s
- Gen Time: 47.69s
- Output Tokens: 2662
- answer read from `answer_tag`

<details><summary>Reasoning (8503 chars, not scored)</summary>

We need answer user's strict formatting. Need produce reference card for four sorting algorithms: insertion sort, merge sort, quicksort, heapsort.
Need order algorithms from best to worst average-case time complexity. Break ties alphabetically.
Average complexities:
- Insertion sort: O(n^2)
- Merge sort: O(n log n)
- Quicksort: O(n log n) average (worst O(n^2))
- Heapsort: O(n log n)
Best to worst average-case: O(n log n) group: heapsort, merge sort, quicksort? Alphabetically: Heapsort, Merge Sort, Quicksort. Then Insertion Sort O(n^2). Need headings Title Case only algorithm's name. Title Case: Heapsort? Merge Sort? Quicksort? Insertion Sort. User says algorithm's name in Title Case. For quicksort maybe Quicksort (one word) or Quick Sort? They listed quicksort. Title Case of quicksort is Quicksort. Merge sort is Merge Sort. Insertion sort is Insertion Sort. Heapsort is Heapsort.
Tables exactly two columns Property Value, exactly four rows in order Best, Average, Worst, Space.
Complexity values O(...) inside backticks.
Need values:
Insertion Sort:
Best O(n) (if already sorted)
Average O(n^2)
Worst O(n^2)
Space O(1)
Merge Sort:
Best O(n log n) (some implementations can be O(n) if already sorted? Standard merge sort best O(n log n) due to divide/conquer; but can be O(n) with optimized? Reference card likely standard: best O(n log n), average O(n log n), worst O(n log n), space O(n). Need choose standard.
Quicksort:
Best O(n log n)
Average O(n log n)
Worst O(n^2)
Space O(log n) (in-place recursion stack)
Heapsort:
Best O(n log n)
Average O(n log n)
Worst O(n log n)
Space O(1)
Need after each table exactly one sentence beginning with "Use it when" — no more than 20 words.
Need ensure sentence starts with Use it when. Count words <=20. Could be:
Heapsort: "Use it when you need guaranteed logarithmic time with constant extra space." Count Use1 it2 when3 you4 need5 guaranteed6 logarithmic7 time8 with9 constant10 extra11 space12. OK.
Merge Sort: "Use it when stable sorting and predictable logarithmic performance matter most." Count Use1 it2 when3 stable4 sorting5 and6 predictable7 logarithmic8 performance9 matter10 most11. OK.
Quicksort: "Use it when average speed and in-place sorting are priorities." Count Use1 it2 when3 average4 speed5 and6 in-place7 sorting8 are9 priorities10. OK.
Insertion Sort: "Use it when inputs are small or nearly sorted." Count Use1 it2 when3 inputs4 are5 small6 or7 nearly8 sorted9. OK.
Need final heading Summary level-3, followed by exactly one bullet list of four items, each form `AlgorithmName — stable` or `AlgorithmName — not stable`.
Stability:
Insertion sort stable.
Merge sort stable.
Quicksort not stable (standard).
Heapsort not stable (standard).
Need order in summary? Probably same as sections? It says four items, each form. Could be same order. Use em dash? They specify `AlgorithmName — stable` with em dash. Need exactly bullet list of four items. Markdown bullets: - Heapsort — not stable etc.
Need output nothing else. Final answer between  tags. Everything inside tags must be answer itself. Need no commentary.
Need ensure headings level-3: ### Heapsort etc.
Need table markdown. Exactly two columns headed Property and Value. Rows exactly four. Values in backticks.
Potential issue: "Order the algorithms from best to worst average-case time complexity. Break ties alphabetically." Alphabetical by algorithm name? Heapsort, Merge Sort, Quicksort. If Title Case names: Heapsort, Merge Sort, Quicksort. Alphabetical ignoring case: Heapsort, Merge Sort, Quicksort. Good.
Need maybe "Quicksort" vs "Quick Sort" alphabetical? If Quick Sort, still after Merge Sort. But user listed quicksort. Use Quicksort.
Need ensure no extra blank lines? It's okay. But "Output nothing else" maybe only answer tags? They ask put complete final answer between  tags. So final should be:

Need exactly one sentence after each table. Blank line between table and sentence? It's okay? They said after each table, exactly one sentence. Markdown may require blank line? Could be table then newline sentence. But if no blank line, markdown table may still render? Usually need blank line after table? In markdown, table can be followed by text on next line? It may be okay but to be safe blank line. Does blank line count as output? It's formatting. Fine.
Need ensure sentence no more than 20 words. Count with punctuation? Words separated by spaces. "Use it when you need guaranteed logarithmic time with constant extra space." 12. Good.
"Use it when stable sorting and predictable logarithmic performance matter most." 11.
"Use it when average speed and in-place sorting are priorities." 10? in-place one word? yes.
"Use it when inputs are small or nearly sorted." 9.
Need maybe "Use it when" exactly first words. Yes.
Need summary bullet list exactly one bullet list of four items. Could be:
### Summary
- Heapsort — not stable
- Merge Sort — stable
- Quicksort — not stable
- Insertion Sort — stable
Need no period.
Need check if "AlgorithmName" should be Title Case? Probably same headings. Use Heapsort, Merge Sort, Quicksort, Insertion Sort.
Need maybe average-case best to worst: Heapsort, Merge Sort, Quicksort all O(n log n). Alphabetical: Heapsort, Merge Sort, Quicksort. Then Insertion Sort.
Need consider if "best" complexity for merge sort could be O(n) if optimized? Standard reference: best O(n log n). If we put O(n log n), average O(n log n), worst O(n log n). Good.
Need consider quicksort space: O(log n) average recursion stack, worst O(n) if unbalanced? But table row Space likely worst-case space? Usually quicksort space O(log n) (in-place). Could be O(log n) for recursion stack. Good.
Need consider heapsort best O(n log n). Yes.
Need ensure all complexity values in form O(...) inside backticks. Use `O(n log n)`, `O(1)`, `O(n^2)`. Good.
Need no other text.
Double-check final answer tags. They want Think first if need, then put complete final answer between . We can just output final with tags. No analysis outside.
Potential hidden issue: "level-3 markdown heading containing only the algorithm's name in Title Case." If we include blank lines, heading line only name. Good.
"Under each heading, a markdown table with exactly two columns headed Property and Value, and exactly four rows in this order: Best, Average, Worst, Space." Table header row plus separator plus four rows. Good.
"After each table, exactly one sentence beginning with the words "Use it when" — no more than 20 words." The sentence should be standalone. Ensure no extra sentence. Good.
"After all four sections, add a final level-3 heading Summary followed by exactly one bullet list of four items" There is no sentence after summary. Good.
Need maybe no blank line between summary heading and bullet list? It's okay. But "followed by exactly one bullet list" maybe immediate next line. We can do:
### Summary
- Heapsort — not stable
...
No blank line? Markdown okay. But if no blank line, still bullet list. Use no blank line to be tight.
Need maybe final answer inside tags no trailing spaces.
Let's craft final.
One more thought: Alphabetical tie: If names Title Case: Heapsort, Merge Sort, Quicksort. If sorting by average complexity best to worst, all same. Good.
Could there be a tie with insertion sort? No.
Need maybe "Merge Sort" average O(n log n), best O(n log n). Some sources best O(n log n). Good.
Need maybe "Insertion Sort" best O(n). Good.
Need maybe "Quicksort" best O(n log n). Good.
Need maybe "Heapsort" best O(n log n). Good.
Need maybe "Space" for merge sort O(n). Good.
Need maybe "Space" for insertion sort O(1). Good.
Need maybe "Space" for quicksort O(log n). Good.
Need maybe "Space" for heapsort O(1). Good.
Check word counts:
Heapsort sentence: Use(1) it(2) when(3) you(4) need(5) guaranteed(6) logarithmic(7) time(8) with(9) constant(10) extra(11) space(12). OK.
Merge: Use1 it2 when3 stable4 sorting5 and6 predictable7 logarithmic8 performance9 matter10 most11. OK.
Quick: Use1 it2 when3 average4 speed5 and6 in-place7 sorting8 are9 priorities10. OK.
Insertion: Use1 it2 when3 inputs4 are5 small6 or7 nearly8 sorted9. OK.
Need maybe "Use it when" exactly, not "Use it when:" no colon. Good.
Need maybe table separator row counts as row? They said exactly four rows in this order: Best, Average, Worst, Space. In markdown table, header row and separator not counted as data rows. Good.
Need maybe no pipe table alignment. Fine.
Need final only answer tags. Ensure no extra newline after </answer>? OK.
</think>

</details>

**Answer (as scored):**

### Heapsort
| Property | Value |
|---|---|
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(1)` |
Use it when you need guaranteed logarithmic time with constant extra space.

### Merge Sort
| Property | Value |
|---|---|
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(n)` |
Use it when stable sorting and predictable logarithmic performance matter most.

### Quicksort
| Property | Value |
|---|---|
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n^2)` |
| Space | `O(log n)` |
Use it when average speed and in-place sorting are priorities.

### Insertion Sort
| Property | Value |
|---|---|
| Best | `O(n)` |
| Average | `O(n^2)` |
| Worst | `O(n^2)` |
| Space | `O(1)` |
Use it when inputs are small or nearly sorted.

### Summary
- Heapsort — not stable
- Merge Sort — stable
- Quicksort — not stable
- Insertion Sort — stable

