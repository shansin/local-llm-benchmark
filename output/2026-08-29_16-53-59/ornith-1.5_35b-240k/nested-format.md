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

- Score: 10.0/10 — All facts correct, ordering and tie-break alphabetical, headings, tables, backticked O(...) values, one Use it when sentence each ≤20 words, Summary bullets correct and no extra text.
- Tokens/s: 206.3
- Prefill tok/s: 2905.8
- TTFT: 0.13s
- Gen Time: 16.39s
- Output Tokens: 3353
- answer read from `answer_tag`

<details><summary>Reasoning (10593 chars, not scored)</summary>

Let me work through this carefully.

First, I need to order the four sorting algorithms from best to worst average-case time complexity, breaking ties alphabetically.

The four algorithms:
- Insertion sort: O(n) best, O(n²) average, O(n²) worst, O(1) space
- Merge sort: O(n log n) best, O(n log n) average, O(n log n) worst, O(n) space
- Quicksort: O(n log n) best, O(n log n) average, O(n²) worst, O(log n) space
- Heapsort: O(n log n) best, O(n log n) average, O(n log n) worst, O(1) space

Average-case time complexities:
- Insertion sort: O(n²)
- Merge sort: O(n log n)
- Quicksort: O(n log n)
- Heapsort: O(n log n)

So merge sort, quicksort, and heapsort all have O(n log n) average case. Insertion sort has O(n²).

Ordering from best to worst:
1. The three with O(n log n): heapsort, merge sort, quicksort (alphabetical)
2. Then insertion sort with O(n²)

So the order is:
1. Heapsort
2. Merge sort
3. Quicksort
4. Insertion sort

Now for each algorithm, I need a level-3 markdown heading with the name in Title Case.

Then a table with two columns: Property and Value, four rows: Best, Average, Worst, Space.

Every complexity value in O(...) inside backticks.

After each table, exactly one sentence beginning with "Use it when" — no more than 20 words.

Then a final level-3 heading Summary followed by a bullet list of four items, each of the form "AlgorithmName — stable" or "AlgorithmName — not stable".

Stability:
- Insertion sort: stable
- Merge sort: stable
- Quicksort: not stable
- Heapsort: not stable

Let me write the complexity values.

Heapsort:
- Best: O(n) — actually heapsort's best case is O(n log n). Let me think. Heapsort always does O(n log n) because it builds a heap and then repeatedly extracts. The best case is O(n log n). Actually, some sources say best case is O(n log n). Let me reconsider.

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
- Space: O(log n)

Insertion sort:
- Best: O(n)
- Average: O(n²)
- Worst: O(n²)
- Space: O(1)

Now the "Use it when" sentences, no more than 20 words each.

Heapsort: "Use it when you need guaranteed worst-case performance with constant extra memory."
Count: Use(1) it(2) when(3) you(4) need(5) guaranteed(6) worst-case(7) performance(8) with(9) constant(10) extra(11) memory(12). 12 words. Good.

Merge sort: "Use it when you need a stable sort with predictable performance."
Count: Use(1) it(2) when(3) you(4) need(5) a(6) stable(7) sort(8) with(9) predictable(10) performance(11). 11 words. Good.

Quicksort: "Use it when average-case speed matters and extra space is limited."
Count: Use(1) it(2) when(3) average-case(4) speed(5) matters(6) and(7) extra(8) space(9) is(10) limited(11). 11 words. Good.

Insertion sort: "Use it for small or nearly sorted datasets where simplicity wins."
Count: Use(1) it(2) for(3) small(4) or(5) nearly(6) sorted(7) datasets(8) where(9) simplicity(10) wins(11). 11 words. Good.

Now the Summary section:
- Heapsort — not stable
- Merge sort — stable
- Quicksort — not stable
- Insertion sort — stable

The order of the bullet list — the instructions don't specify, but I'll follow the same order as the sections.

Let me now format everything.

The heading format: "a level-3 markdown heading containing only the algorithm's name in Title Case."

So:
### Heapsort
### Merge sort
### Quicksort
### Insertion sort

Wait, "Title Case" — "Merge sort" — should it be "Merge Sort"? Title Case typically capitalizes major words. "Merge Sort" would be Title Case. Hmm. But the algorithm is commonly called "Merge sort". Let me think about Title Case.

Title Case capitalizes the first letter of each major word. "Merge Sort" — both words are major, so both capitalized. Actually, for algorithm names, the convention varies. But the instruction says "Title Case", so I should capitalize each word: "Merge Sort", "Quicksort", "Insertion Sort", "Heapsort".

Hmm, but "Quicksort" is one word. "Heapsort" is one word.

Let me reconsider. Title Case:
- Heapsort → Heapsort
- Merge Sort → Merge Sort
- Quicksort → Quicksort
- Insertion Sort → Insertion Sort

Actually, I'll go with Title Case as instructed. So "Merge Sort" and "Insertion Sort".

Wait, but the Summary section says "each of the form `AlgorithmName — stable`". What is AlgorithmName? It should be consistent. Let me use the same names.

Hmm, but actually the stability list — should I use the same naming? Let me be consistent. I'll use Title Case throughout.

Actually, let me reconsider whether "Merge sort" or "Merge Sort". The instruction explicitly says Title Case for the heading. So I'll capitalize both words. For the summary, it says "AlgorithmName" which I interpret as the same name.

Let me finalize:

### Heapsort
| Property | Value |
| --- | --- |
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(1)` |

Use it when you need guaranteed worst-case performance with constant extra memory.

### Merge Sort
| Property | Value |
| --- | --- |
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(n)` |

Use it when you need a stable sort with predictable performance.

### Quicksort
| Property | Value |
| --- | --- |
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n²)` |
| Space | `O(log n)` |

Use it when average-case speed matters and extra space is limited.

### Insertion Sort
| Property | Value |
| --- | --- |
| Best | `O(n)` |
| Average | `O(n²)` |
| Worst | `O(n²)` |
| Space | `O(1)` |

Use it for small or nearly sorted datasets where simplicity wins.

### Summary
- Heapsort — not stable
- Merge Sort — stable
- Quicksort — not stable
- Insertion Sort — stable

Wait, I need to double check the ordering requirement. "Order the algorithms from best to worst average-case time complexity. Break ties alphabetically."

Average-case:
- Heapsort: O(n log n)
- Merge sort: O(n log n)
- Quicksort: O(n log n)
- Insertion sort: O(n²)

Best to worst: O(n log n) is better than O(n²). So the three O(n log n) come first, then insertion sort.

Among the three O(n log n), break ties alphabetically: Heapsort, Merge sort, Quicksort.

So order: Heapsort, Merge sort, Quicksort, Insertion sort. ✓

That's what I have.

Now let me double-check the "Use it when" sentences are each exactly one sentence beginning with "Use it when" and no more than 20 words.

Heapsort: "Use it when you need guaranteed worst-case performance with constant extra memory." — begins with "Use it when", 12 words. ✓

Merge Sort: "Use it when you need a stable sort with predictable performance." — begins with "Use it when", 11 words. ✓

Quicksort: "Use it when average-case speed matters and extra space is limited." — begins with "Use it when", 11 words. ✓

Insertion Sort: "Use it for small or nearly sorted datasets where simplicity wins." — begins with "Use it", not "Use it when"! 

Oops. Let me fix that. It must begin with "Use it when".

Insertion Sort: "Use it when sorting small or nearly sorted data for simplicity."
Count: Use(1) it(2) when(3) sorting(4) small(5) or(6) nearly(7) sorted(8) data(9) for(10) simplicity(11). 11 words. ✓ Begins with "Use it when". ✓

Let me recheck all:
- Heapsort: "Use it when you need guaranteed worst-case performance with constant extra memory." ✓
- Merge Sort: "Use it when you need a stable sort with predictable performance." ✓
- Quicksort: "Use it when average-case speed matters and extra space is limited." ✓
- Insertion Sort: "Use it when sorting small or nearly sorted data for simplicity." ✓

All begin with "Use it when" and are ≤ 20 words.

Now, the table format. The instruction says "a markdown table with exactly two columns headed `Property` and `Value`, and exactly four rows in this order: Best, Average, Worst, Space."

So the header row has Property and Value. Then four data rows. Good.

Let me make sure the complexity values are in O(...) inside backticks. Yes.

Now the Summary: "add a final level-3 heading `Summary` followed by exactly one bullet list of four items, each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`."

So:
### Summary
- Heapsort — not stable
- Merge Sort — stable
- Quicksort — not stable
- Insertion Sort — stable

The dash used is "—" (em dash). The instruction shows "—" so I'll use em dash.

Let me verify stability:
- Heapsort: not stable ✓
- Merge sort: stable ✓
- Quicksort: not stable ✓
- Insertion sort: stable ✓

Everything looks good. Let me also double check that I'm not adding any extra content. No introduction, no conclusion.

One concern: the em dash. The instruction literally shows "—" which is an em dash (U+2014). I'll use that.

Let me reconsider the Title Case issue once more. The instruction says "containing only the algorithm's name in Title Case." Title Case means each major word capitalized. "Merge Sort" and "Insertion Sort" are Title Case. "Heapsort" and "Quicksort" are single words so they stay as is.

Actually, I want to reconsider whether the summary should use the same names. The instruction says "each of the form `AlgorithmName — stable`". AlgorithmName is a placeholder. I'll use the same names as the headings for consistency.

Let me write the final answer.

Actually, let me reconsider the space complexity for quicksort. Quicksort's space is O(log n) for the recursion stack on average, but O(n) worst case. The standard "Space" value quoted is usually O(log n). I'll use O(log n).

And heapsort space is O(1). ✓

Let me also reconsider merge sort space. Standard merge sort is O(n). ✓

Final answer below.

Let me count words once more carefully for each "Use it when" sentence.

Heapsort: "Use it when you need guaranteed worst-case performance with constant extra memory."
Use / it / when / you / need / guaranteed / worst-case / performance / with / constant / extra / memory = 12 words. ✓

Merge Sort: "Use it when you need a stable sort with predictable performance."
Use / it / when / you / need / a / stable / sort / with / predictable / performance = 11 words. ✓

Quicksort: "Use it when average-case speed matters and extra space is limited."
Use / it / when / average-case / speed / matters / and / extra / space / is / limited = 11 words. ✓

Insertion Sort: "Use it when sorting small or nearly sorted data for simplicity."
Use / it / when / sorting / small / or / nearly / sorted / data / for / simplicity = 11 words. ✓

All good.

Now let me produce the final output.
</think>

</details>

**Answer (as scored):**

### Heapsort
| Property | Value |
| --- | --- |
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(1)` |

Use it when you need guaranteed worst-case performance with constant extra memory.

### Merge Sort
| Property | Value |
| --- | --- |
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(n)` |

Use it when you need a stable sort with predictable performance.

### Quicksort
| Property | Value |
| --- | --- |
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n²)` |
| Space | `O(log n)` |

Use it when average-case speed matters and extra space is limited.

### Insertion Sort
| Property | Value |
| --- | --- |
| Best | `O(n)` |
| Average | `O(n²)` |
| Worst | `O(n²)` |
| Space | `O(1)` |

Use it when sorting small or nearly sorted data for simplicity.

### Summary
- Heapsort — not stable
- Merge Sort — stable
- Quicksort — not stable
- Insertion Sort — stable

