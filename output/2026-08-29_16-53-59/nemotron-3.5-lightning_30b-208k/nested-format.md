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

- Score: 9.75/10 — All facts correct and format mostly followed; bullet list uses ' - ' instead of required ' — ' separator, a minor instruction deviation.
- Tokens/s: 288.1
- Prefill tok/s: 2692.4
- TTFT: 29.47s
- Gen Time: 30.34s
- Output Tokens: 8704
- answer read from `answer_tag`

<details><summary>Reasoning (31185 chars, not scored)</summary>

Here's a thinking process:

1.  **Analyze the Request:**
   - Produce a reference card for four sorting algorithms: insertion sort, merge sort, quicksort, heapsort.
   - Format requirements:
     1. Order algorithms from best to worst average-case time complexity. Break ties alphabetically.
     2. For each algorithm, output a level-3 markdown heading containing only the algorithm's name in Title Case.
     3. Under each heading, a markdown table with exactly two columns: `Property` and `Value`, and exactly four rows in order: Best, Average, Worst, Space.
     4. Every complexity value must be written in the form `O(...)` inside backticks.
     5. After each table, exactly one sentence beginning with the words "Use it when" — no more than 20 words.
     6. After all four sections, add a final level-3 heading `Summary` followed by exactly one bullet list of four items, each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`.

   - Output nothing else. No introduction, no conclusion.
   - Put final answer between <answer> and </answer> tags.

2.  **Determine Ordering (Requirement 1):**
   - Need to order from best to worst average-case time complexity.
   - Average-case time complexities:
     - Insertion sort: O(n^2)
     - Merge sort: O(n log n)
     - Quicksort: O(n log n) average
     - Heapsort: O(n log n)
   - So merge sort, quicksort, heapsort all have O(n log n) average. Insertion sort has O(n^2).
   - Break ties alphabetically. The algorithms with O(n log n) average: heapsort, merge sort, quicksort. Alphabetically: heapsort, merge sort, quicksort. Wait, let's check: h, m, q. So order: heapsort, merge sort, quicksort, then insertion sort.
   - But wait: "Order the algorithms from best to worst average-case time complexity. Break ties alphabetically." Best average time means lowest complexity. O(n log n) is better than O(n^2). Among O(n log n), ties broken alphabetically. So: heapsort, merge sort, quicksort, insertion sort.
   - Let me double-check alphabetical order of the three: heapsort, merge sort, quicksort. h < m < q. Yes.

   Wait, is "Title Case" for headings? The heading should contain only the algorithm's name in Title Case. So "Heapsort", "Merge Sort", "Quicksort", "Insertion Sort". But requirement 2 says: "a level-3 markdown heading containing only the algorithm's name in Title Case." Level-3 heading is `### Heading`. Title Case means first letter of each word capitalized. But the algorithm names are typically one word or two words. "Heapsort" is one word, Title Case would be "Heapsort". "Merge Sort" two words, Title Case "Merge Sort". "Quicksort" one word. "Insertion Sort" two words. But wait, the initial list given: "insertion sort, merge sort, quicksort, and heapsort". The requirement says "containing only the algorithm's name in Title Case." I'll output headings as `### Heapsort`, `### Merge Sort`, `### Quicksort`, `### Insertion Sort`. But wait, the ordering is heapsort, merge sort, quicksort, insertion sort. I need to output sections in that order.

   Let me confirm the exact heading format: "a level-3 markdown heading containing only the algorithm's name in Title Case." So `### Heapsort`, etc. I'll make sure the names are exactly in Title Case. "Heapsort", "Merge Sort", "Quicksort", "Insertion Sort".

   Now, for each algorithm, a markdown table with exactly two columns: `Property` and `Value`, and exactly four rows in this order: Best, Average, Worst, Space.

   Complexity values in form `O(...)` inside backticks. So `O(n^2)`, `O(n log n)`, etc. Inside backticks: `` `O(n log n)` ``.

   Rows: Best, Average, Worst, Space. What values for Space? Typically:
   - Insertion sort: Best O(n), Average O(n^2), Worst O(n^2), Space O(1) (in-place)
   - Merge sort: Best O(n log n), Average O(n log n), Worst O(n log n), Space O(n) (not in-place, auxiliary array)
   - Quicksort: Best O(n log n), Average O(n log n), Worst O(n^2), Space O(log n) (due to recursion stack, or O(n) worst-case but typically O(log n) average)
   - Heapsort: Best O(n log n), Average O(n log n), Worst O(n log n), Space O(1) (in-place)

   Let me verify standard complexities:
   - Insertion Sort:
     - Best: O(n) (when array already sorted)
     - Average: O(n^2)
     - Worst: O(n^2)
     - Space: O(1)
   - Merge Sort:
     - Best: O(n log n) (even when sorted, still divides and merges)
     - Average: O(n log n)
     - Worst: O(n log n)
     - Space: O(n) (auxiliary space for merging)
   - Quicksort:
     - Best: O(n log n) (good pivot)
     - Average: O(n log n)
     - Worst: O(n^2) (bad pivot, e.g., already sorted with first pivot)
     - Space: O(log n) (recursion stack, average; worst-case O(n) but typically stated as O(log n) average, or sometimes O(1) if tail-recursion optimized, but standard is O(log n) average space). I'll use O(log n) for space. Some sources say O(n) worst-case, but average space is O(log n). I'll stick with O(log n) as it's commonly cited for quicksort's space complexity.
   - Heapsort:
     - Best: O(n log n)
     - Average: O(n log n)
     - Worst: O(n log n)
     - Space: O(1) (in-place, no significant extra storage)

   Let me double-check heapsort best case. Heapsort always takes O(n log n) even in best case because building heap and extracting always take that much. Yes.

   Now, the "Use it when" sentence after each table, exactly one sentence beginning with "Use it when", no more than 20 words. I need to craft concise sentences.

   After all four sections, a final level-3 heading `Summary` followed by exactly one bullet list of four items, each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`.

   Stability of sorting algorithms:
   - Insertion Sort: stable
   - Merge Sort: stable (if merge is stable, which is standard)
   - Quicksort: not stable (typically)
   - Heapsort: not stable

   So bullet list items: `Heapsort — not stable`, `Insertion Sort — stable`, `Merge Sort — stable`, `Quicksort — not stable`. But wait, the order of the bullet list? Requirement 6 says: "exactly one bullet list of four items, each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`." It doesn't specify order for the bullet list. But probably should match the algorithm order from the sections, or maybe alphabetical? The sections are ordered by average-case complexity. The bullet list might need to be in the same order as the sections, or perhaps it doesn't matter as long as it's four items. I'll order them by the same order as the sections: Heapsort, Merge Sort, Quicksort, Insertion Sort. Or maybe the bullet list should be stable/not stable for each. I'll just list them in the order of the sections: Heapsort, Merge Sort, Quicksort, Insertion Sort. But let's see: "each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`." I'll output: 
   - Heapsort — not stable
   - Merge Sort — stable
   - Quicksort — not stable
   - Insertion Sort — stable

   Wait, the format says "each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`". The dash has spaces around it? It says `AlgorithmName — stable` with an em dash probably, but I'll use `--` or `-`? The example uses ` — ` with spaces around em dash, but in markdown I can just use ` - ` or ` — `. I'll use ` - ` to be safe, or exactly ` — ` as in the prompt. The prompt has `AlgorithmName — stable` with spaces around the dash. I'll emulate that: `Heapsort — not stable`. But in markdown, I need to make sure it renders. I'll just use ` - ` if em dash is tricky, but I can copy the em dash. I'll use ` - ` with spaces, or `—`. The prompt says `AlgorithmName — stable`, so I'll use that exact spacing: word, space, em-dash, space, stable/not stable. I'll write `Heapsort — not stable`. To be safe, I'll use a regular dash with spaces: `Heapsort - not stable`. The prompt uses ` — ` which is em dash with spaces. I'll just use ` - ` or `—`. I think it's fine.

   Now, let's structure the output.

   Order of algorithms: heapsort, merge sort, quicksort, insertion sort. (Based on average-case: O(n log n) three, O(n^2) one; ties broken alphabetically: heapsort, merge sort, quicksort).

   For each, level-3 heading: `### Heapsort`, `### Merge Sort`, `### Quicksort`, `### Insertion Sort`.

   Under each, table with properties Best, Average, Worst, Space.

   Let's fill tables:

   **Heapsort:**
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n log n)` |
   | Space | `O(1)` |

   **Merge Sort:**
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n log n)` |
   | Space | `O(n)` |

   **Quicksort:**
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n^2)` |
   | Space | `O(log n)` |

   **Insertion Sort:**
   | Property | Value |
   | Best | `O(n)` |
   | Average | `O(n^2)` |
   | Worst | `O(n^2)` |
   | Space | `O(1)` |

   Now, "Use it when" sentences, max 20 words. I need exactly one sentence beginning with "Use it when". So the sentence starts with "Use it when", then the rest. Must be no more than 20 words total. Let's count.

   For Heapsort: "Use it when in-place sorting with guaranteed O(n log n) performance is needed." Count: Use(1) it2 when3 in4-place5 sorting6 with7 guaranteed8 O(n9 log10 n11)12 performance13 is14 needed15. That's 15 words. Good.

   Merge Sort: "Use it when stable sorting of large datasets with predictable O(n log n) time is required." Count: Use1 it2 when3 stable4 sorting5 of6 large7 datasets8 with9 predictable10 O(n11 log12 n13)14 time15 is16 required17. 17 words. Good.

   Quicksort: "Use it when fast average-case sorting with low overhead is preferred for in-memory data." Count: Use1 it2 when3 fast4 average-case5 sorting6 with7 low8 overhead9 is10 preferred11 for12 in-memory13 data14. 14 words. Good.

   Insertion Sort: "Use it when sorting small or nearly sorted arrays with minimal overhead." Count: Use1 it2 when3 sorting4 small5 or6 nearly7 sorted8 arrays9 with10 minimal11 overhead12. 12 words. Good.

   Check if any exceed 20. They don't.

   Now, final Summary section: level-3 heading `Summary`, then exactly one bullet list of four items, each `AlgorithmName — stable` or `AlgorithmName — not stable`. I need to decide the order. The sections order was heapsort, merge sort, quicksort, insertion sort. I'll list bullets in that order, or maybe alphabetical? The requirement doesn't specify order for the bullet list, just "exactly one bullet list of four items, each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`." I'll use the same order as the sections: Heapsort, Merge Sort, Quicksort, Insertion Sort. But wait, the algorithm names in Title Case: Heapsort, Merge Sort, Quicksort, Insertion Sort. The bullet items format: `Heapsort — not stable`, `Merge Sort — stable`, `Quicksort — not stable`, `Insertion Sort — stable`. I need to make sure the dash is consistent. I'll use ` - ` with spaces or `—`. The prompt shows `AlgorithmName — stable` with an em dash and spaces. I'll write `Heapsort — not stable`. In markdown, I can just type the em dash or use `--`. I'll use ` - ` to be safe, but the prompt has spaces around the dash. I'll just copy the pattern: `Heapsort — not stable`. I'll ensure there's a space before and after the dash. Actually, the prompt says: "each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`". The backticks show `AlgorithmName — stable`. I'll output exactly that format, but with algorithm names. I'll use ` - ` if em dash causes issues, but in text it's fine. I'll use `—` without spaces? The prompt has spaces. I'll do `Heapsort — not stable`. I can just type `Heapsort - not stable`. I think it's fine.

   Let me re-read requirement 6: "After all four sections, add a final level-3 heading `Summary` followed by exactly one bullet list of four items, each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`." So the bullet list items should be like `- Heapsort — not stable` or just `Heapsort — not stable` as the item. Typically bullet list items start with `- `. I'll do `- Heapsort — not stable`, etc. But the form is `AlgorithmName — stable`. I'll put the algorithm name and the stability status after the dash. I'll write:
   - Heapsort — not stable
   - Merge Sort — stable
   - Quicksort — not stable
   - Insertion Sort — stable

   But wait, the bullet list is "exactly one bullet list of four items". I need to make sure it's a single bullet list, not multiple. I'll do:
   Summary
   - Heapsort — not stable
   - Merge Sort — stable
   - Quicksort — not stable
   - Insertion Sort — stable

   That's four items in one list.

   Now, check all requirements meticulously.

   1. Order algorithms from best to worst average-case time complexity. Break ties alphabetically.
      - Average complexities: Heapsort O(n log n), Merge Sort O(n log n), Quicksort O(n log n), Insertion Sort O(n^2).
      - Best average: O(n log n). Ties broken alphabetically: Heapsort, Merge Sort, Quicksort. Then Insertion Sort.
      - Order: Heapsort, Merge Sort, Quicksort, Insertion Sort. ✓

   2. For each algorithm, output a level-3 markdown heading containing only the algorithm's name in Title Case.
      - `### Heapsort`, `### Merge Sort`, `### Quicksort`, `### Insertion Sort`. ✓ (Title Case: first letter capitalized, rest lower. "Heapsort" is one word, Title Case is Heapsort. "Merge Sort" two words, Title Case Merge Sort. "Quicksort" Quicksort. "Insertion Sort" Insertion Sort. ✓)

   3. Under each heading, a markdown table with exactly two columns headed `Property` and `Value`, and exactly four rows in this order: Best, Average, Worst, Space.
      - I'll have tables with `| Property | Value |` header, then `| Best | ... |` etc. Exactly four rows. ✓

   4. Every complexity value must be written in the form `O(...)` inside backticks.
      - I'll put `O(n log n)`, `O(n^2)`, `O(n)`, `O(log n)` inside backticks: `` `O(n log n)` ``. ✓

   5. After each table, exactly one sentence beginning with the words "Use it when" — no more than 20 words.
      - I'll put after each table a sentence starting with "Use it when". Exactly one sentence. Max 20 words. I counted them: 15, 17, 14, 12. All ≤20. ✓

   6. After all four sections, add a final level-3 heading `Summary` followed by exactly one bullet list of four items, each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`.
      - `### Summary` then bullet list with four items. ✓

   Output nothing else. No introduction, no conclusion. The final answer must be between <answer> and </answer> tags, with only the answer.

   Let me now construct the exact markdown.

   Structure:
   ```
   ### Heapsort
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n log n)` |
   | Space | `O(1)` |
   Use it when in-place sorting with guaranteed O(n log n) performance is needed.

   ### Merge Sort
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n log n)` |
   | Space | `O(n)` |
   Use it when stable sorting of large datasets with predictable O(n log n) time is required.

   ### Quicksort
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n^2)` |
   | Space | `O(log n)` |
   Use it when fast average-case sorting with low overhead is preferred for in-memory data.

   ### Insertion Sort
   | Property | Value |
   | Best | `O(n)` |
   | Average | `O(n^2)` |
   | Worst | `O(n^2)` |
   | Space | `O(1)` |
   Use it when sorting small or nearly sorted arrays with minimal overhead.

   Summary
   - Heapsort — not stable
   - Merge Sort — stable
   - Quicksort — not stable
   - Insertion Sort — stable
   ```

   Wait, check the "Use it when" sentence: it must begin with the words "Use it when". My sentences start with "Use it when". Good. Must be exactly one sentence beginning with those words. I have a period at the end. That's fine.

   Check word counts again carefully:
   - Heapsort: "Use it when in-place sorting with guaranteed O(n log n) performance is needed." 
     Words: Use(1) it(2) when(3) in-place(4) sorting(5) with(6) guaranteed(7) O(n(8) log(9) n)(10) performance(11) is(12) needed(13). Actually, "O(n log n)" is one token? In word count, typically "O(n" might be considered one word, "log" another, "n)" another. But the requirement says "no more than 20 words". I need to ensure the sentence has ≤20 words regardless of how O(...) is counted. I'll count plainly: "Use it when in-place sorting with guaranteed O(n log n) performance is needed." Let's split by spaces: Use, it, when, in-place, sorting, with, guaranteed, O(n, log, n), performance, is, needed. That's 11 or 13 depending on how O(n log n) is tokenized. But to be safe, I'll keep it short. Actually, the backticks might affect word count if spaces are inside, but the sentence is plain text, the backticks are part of the table values, not the sentence. The sentence is separate. I'll make sure the sentence words are counted without the backtick complexities. I'll simplify the sentences to avoid ambiguity.

   Let me rewrite sentences with simple words, avoiding O(...) inside the sentence since that's in the table. The sentence can just describe when to use it without big-O notation, or use it but it's fine. I'll just make sure word count ≤20.

   Revised sentences:
   - Heapsort: "Use it when in-place sorting with guaranteed performance is needed." Words: Use(1) it(2) when(3) in-place(4) sorting(5) with(6) guaranteed(7) performance(8) is(9) needed(10). 10 words.
   - Merge Sort: "Use it when stable sorting of large datasets with predictable performance is required." Words: Use1 it2 when3 stable4 sorting5 of6 large7 datasets8 with9 predictable10 performance11 is12 required13. 13 words.
   - Quicksort: "Use it when fast average-case sorting with low overhead is preferred." Words: Use1 it2 when3 fast4 average-case5 sorting6 with7 low8 overhead9 is10 preferred11. 11 words.
   - Insertion Sort: "Use it when sorting small or nearly sorted arrays with minimal overhead." Words: Use1 it2 when3 sorting4 small5 or6 nearly7 sorted8 arrays9 with10 minimal11 overhead12. 12 words.

   All ≤20. Good.

   But wait, requirement 5: "After each table, exactly one sentence beginning with the words "Use it when" — no more than 20 words." It says "beginning with the words "Use it when"". So the sentence must start with exactly "Use it when". My sentences do. Also, "no more than 20 words". I'll make sure the sentences are exactly starting with "Use it when" and then the rest. I'll put a space after "when". That's fine.

   Now, check the tables. Requirement 3: "a markdown table with exactly two columns headed `Property` and `Value`, and exactly four rows in this order: Best, Average, Worst, Space." I need to make sure the table has exactly those columns and rows. I'll format as:

   ```
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n log n)` |
   | Space | `O(1)` |
   ```

   That's 5 lines including header, 4 data rows. Exactly two columns. ✓

   Requirement 4: "Every complexity value must be written in the form `O(...)` inside backticks." I have `O(n log n)`, etc., inside backticks. ✓

   Now, the order of algorithms in the sections: Heapsort, Merge Sort, Quicksort, Insertion Sort. But wait, the initial list given: "insertion sort, merge sort, quicksort, and heapsort". The requirement 1 says "Order the algorithms from best to worst average-case time complexity. Break ties alphabetically." So I must reorder them as per that rule. I did: Heapsort, Merge Sort, Quicksort, Insertion Sort. But let me double-check alphabetical order of the three with O(n log n) average: heapsort, merge sort, quicksort. h < m < q. Yes. Insertion sort has O(n^2) average, so it's last. So the order is correct.

   But wait: Is "Title Case" for the headings only the algorithm's name, but the algorithm names in the bullet list also need to be in the same case? The bullet list format says `AlgorithmName — stable` or `AlgorithmName — not stable`. It doesn't specify Title Case there, but since the headings are Title Case, I'll use the same names. "Heapsort", "Merge Sort", "Quicksort", "Insertion Sort". In the bullet list, I'll use the same.

   Now, check the Summary bullet list format: "each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`". The example has a space before and after the em dash. I'll use ` — ` with spaces. In markdown, I can just type `Heapsort — not stable`. I need to make sure the dash is rendered as em dash or just a dash. I'll use a regular hyphen with spaces: `Heapsort - not stable`. The prompt uses ` — ` which is em dash, but plain text hyphen with spaces is fine. I'll use ` - ` to be safe, or I can use `—` if I know how. I'll just use ` - ` with spaces around a regular dash. Actually, the prompt has backticks: `AlgorithmName — stable`. The ` — ` might be an em dash. I'll copy the exact pattern but with my algorithm names. I'll write `Heapsort — not stable`. In the final answer, I'll ensure it's clear.

   Let me review the entire output structure and ensure no extra text.

   Potential pitfalls:
   - The "Use it when" sentence must begin with those exact words. I have that.
   - No more than 20 words. I'll count again with exact wording.
   - The tables must have exactly four rows in order: Best, Average, Worst, Space. I have that.
   - Complexity in `O(...)` inside backticks. I have that.
   - Level-3 headings: `### Algorithm Name`. I have `### Heapsort`, etc. But wait, requirement 2: "a level-3 markdown heading containing only the algorithm's name in Title Case." Does "containing only the algorithm's name" mean the heading text should be just the name, no extra characters? Level-3 heading is `### Name`. That's standard. "containing only the algorithm's name" probably means the text after `###` is just the name, not extra words. I have that.
   - After each table, exactly one sentence beginning with "Use it when". I have that, and then a blank line before the next heading? The requirement doesn't forbid blank lines, but it says "Output nothing else. No introduction, no conclusion." I should probably not have blank lines that could be considered extra output, but markdown usually has them for readability. However, to be safe, I might keep them minimal or as per standard. The prompt says "Produce a reference card...". I'll include blank lines between sections for readability, but it might be considered "output nothing else". I think it's fine to have blank lines separating sections, as they are part of the markdown formatting. But to be strictly compliant, I could put them without blank lines, or with. I'll include a single newline between sections, but maybe better to have a blank line for readability. I'll check if the prompt implies no extra commentary, but markdown formatting is expected. I'll include standard markdown spacing.

   Let me re-read: "Output nothing else. No introduction, no conclusion." This means I shouldn't add text like "Here is the reference card:" or "Summary:". The content itself is the answer. Blank lines between sections are part of the markdown structure, not "introduction or conclusion". I'll include them.

   Now, final check of the bullet list in Summary. Requirement 6: "After all four sections, add a final level-3 heading `Summary` followed by exactly one bullet list of four items, each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`." So the heading is `Summary`, level-3, so `### Summary`. Then a bullet list. The bullet list items: I'll use `- Heapsort — not stable`, etc. But the form is `AlgorithmName — stable`. Does the bullet prefix `- ` count as part of the item? Typically, a bullet list item is `- Item`. The item itself should be in the form `AlgorithmName — stable`. So `- Heapsort — not stable` is fine, as the item text is `Heapsort — not stable`. I'll do that.

   Let me ensure the stability designations are correct:
   - Heapsort: not stable (correct, because heapify can change relative order of equal elements)
   - Merge Sort: stable (standard merge is stable)
   - Quicksort: not stable (partitioning is not stable)
   - Insertion Sort: stable (insertion maintains order of equal keys)

   All good.

   Now, let me draft the exact markdown in my mind, ensuring all constraints.

   ```
   ### Heapsort
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n log n)` |
   | Space | `O(1)` |
   Use it when in-place sorting with guaranteed performance is needed.

   ### Merge Sort
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n log n)` |
   | Space | `O(n)` |
   Use it when stable sorting of large datasets with predictable performance is required.

   ### Quicksort
   | Property | Value |
   | Best | `O(n log n)` |
   | Average | `O(n log n)` |
   | Worst | `O(n^2)` |
   | Space | `O(log n)` |
   Use it when fast average-case sorting with low overhead is preferred.

   ### Insertion Sort
   | Property | Value |
   | Best | `O(n)` |
   | Average | `O(n^2)` |
   | Worst | `O(n^2)` |
   | Space | `O(1)` |
   Use it when sorting small or nearly sorted arrays with minimal overhead.

   ### Summary
   - Heapsort — not stable
   - Merge Sort — stable
   - Quicksort — not stable
   - Insertion Sort — stable
   ```

   Wait, check the "Use it when" sentences word counts again with the exact text:

   1. "Use it when in-place sorting with guaranteed performance is needed."
      Words: Use(1) it(2) when(3) in-place(4) sorting(5) with(6) guaranteed(7) performance(8) is(9) needed(10). 10 words. ✓

   2. "Use it when stable sorting of large datasets with predictable performance is required."
      Words: Use1 it2 when3 stable4 sorting5 of6 large7 datasets8 with9 predictable10 performance11 is12 required13. 13 words. ✓

   3. "Use it when fast average-case sorting with low overhead is preferred."
      Words: Use1 it2 when3 fast4 average-case5 sorting6 with7 low8 overhead9 is10 preferred11. 11 words. ✓

   4. "Use it when sorting small or nearly sorted arrays with minimal overhead."
      Words: Use1 it2 when3 sorting4 small5 or6 nearly7 sorted8 arrays9 with10 minimal11 overhead12. 12 words. ✓

   All ≤20. Good.

   One thing: The sentences must "begin with the words 'Use it when'". My sentences start with "Use it when". The next word is "in-place", "stable", "fast", "sorting". That's fine.

   Now, check if the tables have exactly two columns headed `Property` and `Value`. Yes.

   Exactly four rows in order: Best, Average, Worst, Space. Yes.

   Complexity values in form `O(...)` inside backticks. Yes.

   Level-3 headings containing only the algorithm's name in Title Case. `### Heapsort` - the name is "Heapsort", Title Case. `### Merge Sort` - "Merge Sort" is Title Case. `### Quicksort` - "Quicksort". `### Insertion Sort` - "Insertion Sort". All good.

   Order of algorithms: Heapsort, Merge Sort, Quicksort, Insertion Sort. Based on average-case: O(n log n), O(n log n), O(n log n), O(n^2). Ties broken alphabetically: heapsort, merge sort, quicksort. Correct.

   Summary bullet list: four items, each `AlgorithmName — stable` or `not stable`. I have Heapsort — not stable, Merge Sort — stable, Quicksort — not stable, Insertion Sort — stable. The format uses ` — ` with spaces. I'll make sure there's a space before and after the dash. In my draft, I have `Heapsort — not stable`. That's fine.

   One potential issue: The requirement says "Break ties alphabetically." Does that mean the order of the algorithms in the output should be from best to worst average-case, with ties broken alphabetically? Yes. I have that.

   Another thing: "For each algorithm, output a level-3 markdown heading containing only the algorithm's name in Title Case." Does "only the algorithm's name" mean the heading text should be just the name, e.g., `### Heapsort` and not `### Heapsort Sorting Algorithm`? I have just the name. Good.

   Now, I'll produce the final answer inside `<answer>` and `</answer>`. I need to make sure there's no extra text commentary. I'll just put the markdown content.

   Let me review the entire output for any possible violations.

   - Order: Heapsort, Merge Sort, Quicksort, Insertion Sort. ✓
   - Headings: `### Heapsort`, etc. ✓
   - Tables: 2 columns, 4 rows, Best/Average/Worst/Space. ✓
   - Complexities in `O(...)` backticks. ✓
   - "Use it when" sentences after each table, ≤20 words, beginning with those words. ✓
   - Summary section with level-3 heading `### Summary` and bullet list of four items in the specified form. ✓
   - No introduction, no conclusion. ✓

   One minor thing: The "Use it when" sentence must be "exactly one sentence beginning with the words 'Use it when' — no more than 20 words." Does "beginning with the words" mean the sentence must start with those exact words, and nothing before? Yes. My sentences start with "Use it when". Good.

   Also, "After each table, exactly one sentence beginning with the words 'Use it when'". So after each table, I place that sentence. I have a blank line then the next heading? The sentence is after the table, then a blank line, then the next heading. That's fine. The sentence is after the table, before the next heading. I'll structure it as:

   ```
   ### Heapsort
   table
   Use it when ...
   
   ### Merge Sort
   table
   Use it when ...
   ```

   That's standard.

   Now, final check of the bullet list items format. The requirement says: "each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`". The backticks in the prompt show `AlgorithmName — stable`. I'll output without backticks, just the text. I'll use `Heapsort — not stable`. The dash has spaces around it. I'll ensure there's a space before and after the dash. In markdown, `Heapsort — not stable` will render with an em dash if the system supports it, or I can use `Heapsort - not stable`. I'll use ` - ` with regular hyphen and spaces, as it's safer. But the prompt uses ` — ` which is em dash. I'll just copy the exact string ` — ` but in my text I'll use ` - ` or `—`. Actually, I can just write `Heapsort - not stable`. The spaces around the dash match the prompt's ` — ` if I use em dash. I'll use `—` without spaces? The prompt has spaces. I'll do `Heapsort — not stable`. I'll type it with an em dash. In pure text, I can use `--` but with spaces. I'll just use ` - ` (space, hyphen, space). To be precise, I'll write `Heapsort - not stable`. The prompt example: `AlgorithmName — stable`. I'll mimic `Heapsort — stable` but with my words. I think it's fine.

   Let me just use `Heapsort - not stable` etc. It's clear.

   Wait, requirement 6: "each of the form `AlgorithmName — stable` or `AlgorithmName — not stable`". The form includes the dash with spaces. I'll output exactly that: `Heapsort — not stable`. I'll make sure there's a non-breaking space or just regular space. I'll use regular space.

   Now, I'll produce the final answer.

</details>

**Answer (as scored):**

### Heapsort
| Property | Value |
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(1)` |
Use it when in-place sorting with guaranteed performance is needed.

### Merge Sort
| Property | Value |
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n log n)` |
| Space | `O(n)` |
Use it when stable sorting of large datasets with predictable performance is required.

### Quicksort
| Property | Value |
| Best | `O(n log n)` |
| Average | `O(n log n)` |
| Worst | `O(n^2)` |
| Space | `O(log n)` |
Use it when fast average-case sorting with low overhead is preferred.

### Insertion Sort
| Property | Value |
| Best | `O(n)` |
| Average | `O(n^2)` |
| Worst | `O(n^2)` |
| Space | `O(1)` |
Use it when sorting small or nearly sorted arrays with minimal overhead.

### Summary
- Heapsort - not stable
- Merge Sort - stable
- Quicksort - not stable
- Insertion Sort - stable

