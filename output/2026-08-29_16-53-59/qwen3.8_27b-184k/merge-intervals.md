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

- Score: 8.5/10 — Function is correct with type hints and 5 tests covering edge cases, but automated check reports no valid Python found, indicating a formatting/parsing issue that reduces instruction following.
- Tokens/s: 106.4
- Prefill tok/s: 997.6
- TTFT: 6.47s
- Gen Time: 13.94s
- Output Tokens: 1455
- answer read from `answer_tag`

<details><summary>Reasoning (2025 chars, not scored)</summary>

The user wants a Python function `merge_intervals` that:
1. Takes a list of intervals (tuples of two integers)
2. Merges all overlapping intervals
3. Returns merged intervals sorted by start time
4. Handles edge cases: empty list, single interval, already sorted, unsorted input
5. Time complexity O(n log n)
6. Includes type hints
7. Includes 5 test cases covering edge cases

Let me think about the algorithm:
1. Sort intervals by start time - O(n log n)
2. Iterate through sorted intervals, merging overlapping ones - O(n)
3. Two intervals overlap if the start of the current interval is <= the end of the previous merged interval

Algorithm:
- If list is empty, return empty list
- Sort by start time
- Initialize result with first interval
- For each subsequent interval:
  - If it overlaps with the last interval in result (current.start <= last.end), merge them (update last.end = max(last.end, current.end))
  - Otherwise, append to result
- Return result

Let me write this with type hints:

```python
from typing import List, Tuple

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged
```

Now for 5 test cases:
1. Empty list: [] => []
2. Single interval: [(1, 5)] => [(1, 5)]
3. Overlapping intervals (the example): [(1,3), (2,6), (8,10), (15,18)] => [(1,6), (8,10), (15,18)]
4. Unsorted input: [(8,10), (1,3), (15,18), (2,6)] => [(1,6), (8,10), (15,18)]
5. All overlapping: [(1,10), (2,5), (3,7)] => [(1,10)]

Let me also consider:
- Nested intervals: [(1,10), (2,3)] => [(1,10)]
- Adjacent intervals: [(1,2), (2,3)] - these should merge since 2 <= 2

Let me finalize the code with proper test cases.

</details>

**Answer (as scored):**

```python
from typing import List, Tuple


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge all overlapping intervals and return them sorted by start time.
    
    Args:
        intervals: A list of tuples, each representing an interval [start, end].
        
    Returns:
        A list of merged non-overlapping intervals sorted by start time.
        
    Time Complexity: O(n log n) due to sorting.
    Space Complexity: O(n) for the sorted list and merged result.
    """
    # Edge case: empty list
    if not intervals:
        return []
    
    # Edge case: single interval
    if len(intervals) == 1:
        return [intervals[0]]
    
    # Sort intervals by start time - O(n log n)
    sorted_intervals: List[Tuple[int, int]] = sorted(intervals, key=lambda x: x[0])
    
    # Initialize merged list with the first interval
    merged: List[Tuple[int, int]] = [sorted_intervals[0]]
    
    # Iterate through remaining intervals and merge if overlapping - O(n)
    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        
        # If current interval overlaps with the last merged interval
        if current_start <= last_end:
            # Merge: keep the earlier start, take the max of the ends
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap: add as a new interval
            merged.append((current_start, current_end))
    
    return merged


# ============ TEST CASES ============

if __name__ == "__main__":
    # Test 1: Empty list
    assert merge_intervals([]) == [], "Test 1 Failed: Empty list"
    print("Test 1 Passed: Empty list => []")
    
    # Test 2: Single interval
    assert merge_intervals([(5, 10)]) == [(5, 10)], "Test 2 Failed: Single interval"
    print("Test 2 Passed: Single interval => [(5, 10)]")
    
    # Test 3: Overlapping intervals (standard example)
    result = merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)])
    assert result == [(1, 6), (8, 10), (15, 18)], f"Test 3 Failed: Got {result}"
    print("Test 3 Passed: Overlapping intervals => [(1, 6), (8, 10), (15, 18)]")
    
    # Test 4: Unsorted input
    result = merge_intervals([(8, 10), (1, 3), (15, 18), (2, 6)])
    assert result == [(1, 6), (8, 10), (15, 18)], f"Test 4 Failed: Got {result}"
    print("Test 4 Passed: Unsorted input => [(1, 6), (8, 10), (15, 18)]")
    
    # Test 5: All intervals overlap (nested intervals)
    result = merge_intervals([(1, 10), (2, 5), (3, 7), (4, 6)])
    assert result == [(1, 10)], f"Test 5 Failed: Got {result}"
    print("Test 5 Passed: All overlapping/nested => [(1, 10)]")
    
    print("\nAll 5 test cases passed!")

