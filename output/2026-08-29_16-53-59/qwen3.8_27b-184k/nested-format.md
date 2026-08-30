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

- Score: 10.0/10 — All facts correct, ordering alphabetical for O(n log n) tie, headings, tables, backticked O(...) values, one Use it when sentence each ≤20 words, Summary bullets correct, no extra text.
- Tokens/s: 106.9
- Prefill tok/s: 1155.1
- TTFT: 15.02s
- Gen Time: 17.72s
- Output Tokens: 1855
- answer read from `answer_tag`

<details><summary>Reasoning (4811 chars, not scored)</summary>

Let me work through this carefully.

**Step 1: Determine average-case time complexity for each algorithm.**

- Insertion sort: O(n²)
- Merge sort: O(n log n)
- Quicksort: O(n log n)
- Heapsort: O(n log n)

**Step 2: Order from best to worst average-case time complexity. Break ties alphabetically.**

Best: O(n log n) — Heapsort, Merge sort, Quicksort (alphabetical: Heapsort, Merge sort, Quicksort)
Worst: O(n²) — Insertion sort

So the order is:
1. Heapsort
2. Merge sort
3. Quicksort
4. Insertion sort

**Step 3: Fill in the properties for each.**

Heapsort:
- Best: O(n log n)
- Average: O(n log n)
- Worst: O(n log n)
- Space: O(1)

Merge sort:
- Best: O(n log n)
- Average: O(n log n)
- Worst: O(n log n)
- Space: O(n)

Quicksort:
- Best: O(n log n)
- Average: O(n log n)
- Worst: O(n²)
- Space: O(log n) — for the recursion stack

Insertion sort:
- Best: O(n)
- Average: O(n²)
- Worst: O(n²)
- Space: O(1)

**Step 4: Format each section.**

Level-3 heading with algorithm name in Title Case:
- Heapsort
- Merge Sort (Title Case)
- Quicksort
- Insertion Sort (Title Case)

Wait, "Title Case" — let me think about this. "Merge sort" in Title Case would be "Merge Sort". "Insertion sort" in Title Case would be "Insertion Sort". "Quicksort" is one word so it's "Quicksort". "Heapsort" is one word so it's "Heapsort".

**Step 5: "Use it when" sentences — no more than 20 words, beginning with "Use it when".**

Heapsort: "Use it when you need guaranteed O(n log n) performance with minimal extra memory." — Let me count: Use(1) it(2) when(3) you(4) need(5) guaranteed(6) O(n(7) log(8) n)(9) performance(10) with(11) minimal(12) extra(13) memory(14). That's 14 words. Wait, is "O(n log n)" one word or multiple? Let me count more carefully. I'll treat each space-separated token as a word.

"Use it when you need guaranteed O(n log n) performance with minimal extra memory."
Use(1) it(2) when(3) you(4) need(5) guaranteed(6) O(n(7) log(8) n)(9) performance(10) with(11) minimal(12) extra(13) memory(14). 14 words. Good.

Merge Sort: "Use it when you need stable sorting and can afford extra memory."
Use(1) it(2) when(3) you(4) need(5) stable(6) sorting(7) and(8) can(9) afford(10) extra(11) memory(12). 12 words. Good.

Quicksort: "Use it when average performance matters and input is unlikely to be adversarial."
Use(1) it(2) when(3) average(4) performance(5) matters(6) and(7) input(8) is(9) unlikely(10) to(11) be(12) adversarial(13). 13 words. Good.

Insertion Sort: "Use it when the dataset is small or nearly sorted already."
Use(1) it(2) when(3) the(4) dataset(5) is(6) small(7) or(8) nearly(9) sorted(10) already(11). 11 words. Good.

**Step 6: Summary section.**

Stability:
- Heapsort: not stable
- Merge Sort: stable
- Quicksort: not stable
- Insertion Sort: stable

Format: `AlgorithmName — stable` or `AlgorithmName — not stable`

The bullet list should have four items. The order should match the order of the algorithms above (Heapsort, Merge Sort, Quicksort, Insertion Sort).

- Heapsort — not stable
- Merge Sort — stable
- Quicksort — not stable
- Insertion Sort — stable

**Step 7: Check all format requirements.**

1. ✓ Order: Heapsort, Merge Sort, Quicksort, Insertion Sort (best to worst avg, ties alphabetical)
2. ✓ Level-3 headings with Title Case names
3. ✓ Tables with Property/Value columns, four rows: Best, Average, Worst, Space
4. ✓ Complexity values in O(...) inside backticks
5. ✓ One sentence starting with "Use it when", ≤20 words
6. ✓ Summary heading with bullet list of four items

Let me also double-check: "Output nothing else. No introduction, no conclusion."

Let me also verify the space complexity for Quicksort. The in-place version uses O(log n) stack space on average. I'll use O(log n).

Wait, for the "Use it when" sentences, the requirement says "exactly one sentence beginning with the words 'Use it when'". Let me make sure each is exactly one sentence.

Let me also re-examine: the requirement says "no more than 20 words." Let me recount more carefully.

Heapsort: "Use it when you need guaranteed O(n log n) performance with minimal extra memory."
Words: Use, it, when, you, need, guaranteed, O(n, log, n), performance, with, minimal, extra, memory = 14 words. ✓

Merge Sort: "Use it when you need stable sorting and can afford extra memory."
Words: Use, it, when, you, need, stable, sorting, and, can, afford, extra, memory = 12 words. ✓

Quicksort: "Use it when average performance matters and input is unlikely to be adversarial."
Words: Use, it, when, average, performance, matters, and, input, is, unlikely, to, be, adversarial = 13 words. ✓

Insertion Sort: "Use it when the dataset is small or nearly sorted already."
Words: Use, it, when, the, dataset, is, small, or, nearly, sorted, already = 11 words. ✓

All good. Let me write the final answer.

</details>

**Answer (as scored):**

### Heapsort

| Property | Value |
|----------|-------|
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(1)` |

Use it when you need guaranteed O(n log n) performance with minimal extra memory.

### Merge Sort

| Property | Value |
|----------|-------|
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(n)` |

Use it when you need stable sorting and can afford extra memory.

### Quicksort

| Property | Value |
|----------|-------|
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n²)` |
| Space | `O(log n)` |

Use it when average performance matters and input is unlikely to be adversarial.

### Insertion Sort

| Property | Value |
|----------|-------|
| Best | `O(n)` |
| Average | `O(n²)` |
| Worst | `O(n²)` |
| Space | `O(1)` |

Use it when the dataset is small or nearly sorted already.

### Summary

- Heapsort — not stable
- Merge Sort — stable
- Quicksort — not stable
- Insertion Sort — stable

