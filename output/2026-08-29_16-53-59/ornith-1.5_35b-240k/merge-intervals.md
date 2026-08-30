# coding / merge-intervals

**Prompt:** Write a Python function called `merge_intervals` that takes a list of intervals (each represented as a tuple of two integers [start, end]) and merges all overlapping intervals. The function should return the merged intervals sorted by start time.

Requirements:
- Handle edge cases: empty list, single interval, already sorted, unsorted input
- Time complexity should be O(n log n)
- Include type hints
- Include 5 test cases that cover edge cases

Example:
  merge_intervals([(1,3), (2,6), (8,10), (15,18)]) => [(1,6), (8,10), (15,18)]

## Repeat 1 (seed 0)

- Score: 9.75/10 — Correct sorting-merge algorithm with type hints, handles all edge cases, 5+ tests, and passes all automated checks; clarity slightly reduced by adjacent-interval merge semantics but acceptable.
- Tokens/s: 208.9
- Prefill tok/s: 2122.8
- TTFT: 0.13s
- Gen Time: 8.98s
- Output Tokens: 1849
- answer read from `answer_tag`

<details><summary>Reasoning (4071 chars, not scored)</summary>

The user wants a Python function `merge_intervals` that merges overlapping intervals. Let me write this carefully.

The algorithm:
1. Sort intervals by start time
2. Iterate through, merging overlapping intervals
3. Return merged intervals

Let me think about the edge cases:
- Empty list: return []
- Single interval: return [interval]
- Already sorted: works
- Unsorted input: sort first

Time complexity: O(n log n) due to sorting.

Let me write the code with type hints and 5 test cases.

Let me think about the merging logic:
- Sort by start
- Initialize result with first interval
- For each subsequent interval:
  - If it overlaps with the last interval in result (start <= last end), merge by updating end to max
  - Otherwise, add as new interval

Let me write this:

```python
from typing import List, Tuple

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [sorted_intervals[0]]
    
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping, merge
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged
```

Now for the test cases:

1. Empty list: `merge_intervals([]) == []`
2. Single interval: `merge_intervals([(1, 5)]) == [(1, 5)]`
3. Already sorted with overlaps: `merge_intervals([(1,3), (2,6), (8,10), (15,18)]) == [(1,6), (8,10), (15,18)]`
4. Unsorted input: `merge_intervals([(15,18), (1,3), (8,10), (2,6)]) == [(1,6), (8,10), (15,18)]`
5. No overlaps: `merge_intervals([(1,2), (3,4), (5,6)]) == [(1,2), (3,4), (5,6)]`

Let me also add a test for nested intervals where one contains another:
`merge_intervals([(1,10), (2,5)]) == [(1,10)]`

Let me write the complete answer with the test cases.

Let me verify the example:
- Input: [(1,3), (2,6), (8,10), (15,18)]
- Sorted: [(1,3), (2,6), (8,10), (15,18)]
- merged = [(1,3)]
- (2,6): 2 <= 3, merge -> (1, max(3,6)) = (1,6). merged = [(1,6)]
- (8,10): 8 > 6, add. merged = [(1,6), (8,10)]
- (15,18): 15 > 10, add. merged = [(1,6), (8,10), (15,18)]
- Result: [(1,6), (8,10), (15,18)] ✓

Now let me write the final answer. I'll include a main block with the test cases.

Let me make sure the test cases are comprehensive and cover all the edge cases mentioned.

I'll write it cleanly.
</think>

```python
from typing import List, Tuple


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping intervals.

    Args:
        intervals: A list of intervals, each represented as a (start, end) tuple.

    Returns:
        A list of merged intervals sorted by start time.
    """
    if not intervals:
        return []

    # Sort by start time -> O(n log n)
    sorted_intervals = sorted(intervals, key=lambda interval: interval[0])

    merged: List[Tuple[int, int]] = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping (or adjacent) intervals -> merge by extending the end.
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


if __name__ == "__main__":
    # Test 1: Empty list
    assert merge_intervals([]) == []

    # Test 2: Single interval
    assert merge_intervals([(1, 5)]) == [(1, 5)]

    # Test 3: Already sorted input with overlaps
    assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]

    # Test 4: Unsorted input with overlaps
    assert merge_intervals([(15, 18), (1, 3), (8, 10), (2, 6)]) == [(1, 6), (8, 10), (15, 18)]

    # Test 5: No overlaps (and a nested interval case)
    assert merge_intervals([(1, 2), (3, 4), (5, 6)]) == [(1, 2), (3, 4), (5, 6)]
    assert merge_intervals([(1, 10), (2, 5)]) == [(1, 10)]

    print("All test cases passed.")
```

</details>

**Answer (as scored):**

```python
from typing import List, Tuple


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping intervals.

    Args:
        intervals: A list of intervals, each represented as a (start, end) tuple.

    Returns:
        A list of merged intervals sorted by start time.
    """
    if not intervals:
        return []

    # Sort by start time -> O(n log n)
    sorted_intervals = sorted(intervals, key=lambda interval: interval[0])

    merged: List[Tuple[int, int]] = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping (or adjacent) intervals -> merge by extending the end.
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


if __name__ == "__main__":
    # Test 1: Empty list
    assert merge_intervals([]) == []

    # Test 2: Single interval
    assert merge_intervals([(1, 5)]) == [(1, 5)]

    # Test 3: Already sorted input with overlaps
    assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]

    # Test 4: Unsorted input with overlaps
    assert merge_intervals([(15, 18), (1, 3), (8, 10), (2, 6)]) == [(1, 6), (8, 10), (15, 18)]

    # Test 5: No overlaps (and a nested interval case)
    assert merge_intervals([(1, 2), (3, 4), (5, 6)]) == [(1, 2), (3, 4), (5, 6)]
    assert merge_intervals([(1, 10), (2, 5)]) == [(1, 10)]

    print("All test cases passed.")
```

