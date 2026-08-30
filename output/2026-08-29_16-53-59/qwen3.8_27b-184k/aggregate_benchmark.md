# Benchmark: qwen3.8:27b-184k

- **Parameters:** 27.3B
- **Quantization:** Q4_K_M
- **Family:** qwen35
- **Cold load:** 9.90s

## Throughput (fixed probe)

| Tokens/s | IQR | Prefill tok/s | TTFT p50 (s) | TTFT p90 (s) | Samples |
|----------|-----|---------------|--------------|--------------|---------|
| 86.1 | 5.8 | 1264.7 | 2.78 | 2.92 | 5 |

## Per-task measurements

| Category | Task | Difficulty | Tokens/s | Gen Time (s) | Tokens | Score |
|----------|------|------------|----------|--------------|--------|-------|
| coding | balanced-brackets | easy | 99.4 | 7.39 | 674 | 10.0 |
| coding | fix-date-overlap | hard | 104.8 | 6.74 | 673 | 10.0 |
| coding | fix-run-length | medium | 108.4 | 8.74 | 914 | 10.0 |
| coding | fix-window-sum | medium | 99.5 | 14.71 | 1432 | 10.0 |
| coding | lru-cache | hard | 96.2 | 23.39 | 2222 | 9.8 |
| coding | merge-intervals | medium | 106.4 | 13.94 | 1455 | 8.5 |
| coding | parse-logs | medium | 93.2 | 29.73 | 2743 | 9.8 |
| coding | topological-sort | hard | 95.1 | 20.19 | 1891 | 10.0 |
| faithfulness | false-premises | medium | 111.1 | 11.08 | 1189 | 10.0 |
| faithfulness | overclaimed-evidence | hard | 103.1 | 8.69 | 862 | 10.0 |
| faithfulness | unanswerable-questions | hard | 98.9 | 8.20 | 772 | 9.5 |
| faithfulness | unusual-semantics | hard | 111.6 | 6.36 | 652 | 10.0 |
| instruction | chained-conversions | medium | 95.3 | 20.97 | 1967 | 9.2 |
| instruction | constraint-precedence | hard | 105.6 | 17.41 | 1811 | 1.0 |
| instruction | impossible-constraints | hard | 103.8 | 2.66 | 249 | 10.0 |
| instruction | lipogram | hard | 119.0 | 18.56 | 2181 | 1.0 |
| instruction | negative-constraints | easy | 102.8 | 24.74 | 2518 | 1.0 |
| instruction | nested-format | hard | 106.9 | 17.72 | 1855 | 10.0 |
| instruction | strict-json | medium | 115.3 | 4.24 | 453 | 10.0 |
| knowledge | antibiotic-resistance | medium | 86.9 | 45.78 | 3958 | 9.0 |
| knowledge | attention-mechanism | medium | 80.5 | 17.10 | 1354 | 9.0 |
| knowledge | fission-fusion | medium | 75.7 | 15.85 | 1183 | 9.0 |
| knowledge | floating-point | hard | 82.6 | 20.14 | 1641 | 9.0 |
| knowledge | tcp-vs-udp | easy | 70.5 | 12.36 | 855 | 9.0 |
| longcontext | ledger-audit | hard | 111.5 | 108.33 | 11011 | 10.0 |
| longcontext | policy-lookup | hard | 107.2 | 10.85 | 643 | 9.5 |
| longcontext | scattered-facts | hard | 117.3 | 5.78 | 306 | 10.0 |
| reasoning | knights-knaves | medium | 102.6 | 8.86 | 877 | 10.0 |
| reasoning | schedule-constraints | medium | 106.7 | 9.89 | 1028 | 9.0 |
| reasoning | twelve-balls | hard | 112.1 | 47.24 | 5268 | 9.8 |
| reasoning | work-rate | easy | 108.9 | 6.40 | 666 | 5.8 |
| reasoning | zebra-puzzle | medium | 93.8 | 205.34 | 19238 | 3.2 |
| statetrack | ledger-balance | medium | 117.8 | 19.43 | 2228 | 10.0 |
| statetrack | robot-grid | medium | 116.8 | 24.83 | 2859 | 8.2 |
| statetrack | stack-machine | hard | 121.2 | 9.91 | 1148 | 10.0 |
| transformation | computed-table | medium | 117.2 | 19.19 | 2184 | 10.0 |
| transformation | csv-to-json | medium | 116.6 | 17.87 | 1991 | 10.0 |
| transformation | schema-migration | hard | 121.6 | 15.72 | 1830 | 10.0 |
| writing | dialogue-only | hard | 99.4 | 117.55 | 11645 | 1.0 |
| writing | explain-to-child | easy | 93.1 | 33.62 | 3101 | 8.5 |
| writing | lighthouse | medium | 64.6 | 18.51 | 1181 | 9.0 |
| writing | technical-to-plain | medium | 108.5 | 101.68 | 10996 | 8.8 |
| writing | unreliable-narrator | hard | 94.5 | 66.92 | 6289 | 9.0 |

## Responses

### coding / balanced-brackets

**Prompt:** Write a Python function called `is_balanced` that takes a string and returns True if all brackets in it are correctly balanced and nested, and False otherwise.

Consider three bracket types: `()`, `[]...

**Response:**

<answer>
```python
def is_balanced(s: str) -> bool:
    """
    Check if all brackets in the string are correctly balanced and nested.
    
    Supports three bracket types: (), [], {}
    All other characters are ignored.
    
    Args:
        s: The input string to check.
        
    Returns:
        True if all brackets are balanced and properly nested, False otherwise.
    """
    stack: list[str] = []
    matching_close: dict[str, str] = {')': '(', ']': '[', '}': '{'}
    opening: set[str] = {'(', '[', '{'}

    for char in s:
        if char in opening:
            stack.append(char)
        elif char in matching_close:
            if not stack or stack[-1] != matching_close[char]:
                return False
            stack.pop()
        # All other characters are ignored

    return len(stack) == 0
```
</answer>

---

### coding / fix-date-overlap

**Prompt:** The function below is supposed to return the number of days two date ranges have in common.
Both ranges are inclusive of their end dates: a booking from the 3rd to the 5th covers three
days.

```pytho...

**Response:**

<answer>
```python
from datetime import date


def overlap_days(a_start: date, a_end: date, b_start: date, b_end: date) -> int:
    """Days covered by both ranges. Both ranges include their end date."""
    if a_start > a_end or b_start > b_end:
        raise ValueError("each range must start on or before it ends")

    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if start > end:
        return 0
    return (end - start).days + 1
```

The only bug was in the return statement: `(end - start).days` computes the *difference* between two dates, which for an inclusive range is always one less than the actual number of days covered (e.g., 3rd→5th gives 2 instead of 3). Adding `+ 1` accounts for the inclusive end date.
</answer>

---

### coding / fix-run-length

**Prompt:** The function below is supposed to run-length encode a string: return a list of
`(character, count)` pairs, one per run of consecutive identical characters, in order.

```python
def encode(data: str) -...

**Response:**

<answer>
```python
def encode(data: str) -> list[tuple[str, int]]:
    """Run-length encode a string.

    encode("aaabbc") => [("a", 3), ("b", 2), ("c", 1)]
    """
    out: list[tuple[str, int]] = []
    if not data:
        return out

    current = data[0]
    count = 1
    for ch in data[1:]:
        if ch == current:
            count += 1
        else:
            out.append((current, count))
            current = ch
            count = 1
    out.append((current, count))
    return out
```

Two bugs: (1) the loop iterated over the entire string including the first character, which was already counted in the initial `count = 1`, so the first run was always off by one; (2) the final run was never appended to `out` after the loop finished.
</answer>

---

### coding / fix-window-sum

**Prompt:** The function below is supposed to return the maximum sum of any contiguous window of exactly
`k` elements in `values`. It is subtly wrong: it passes on some inputs and fails on others.

```python
def ...

**Response:**

<answer>
```python
def max_window_sum(values: list[int], k: int) -> int:
    """Return the largest sum of any k consecutive elements."""
    if k <= 0 or k > len(values):
        raise ValueError("k must be between 1 and len(values)")

    best = sum(values[:k])
    window = best
    for i in range(k, len(values)):
        window += values[i] - values[i - k]
        if window > best:
            best = window
    return best
```

`best` was initialized to `0` instead of the first window's sum, so the first window was never compared and any all-negative input would incorrectly return `0`. Initializing `best` to `sum(values[:k])` fixes both issues in one change.
</answer>

---

### coding / lru-cache

**Prompt:** Implement a Python class called `LRUCache` with a fixed capacity that evicts the least recently used entry when full.

API:
- `LRUCache(capacity: int)` — create a cache holding at most `capacity` item...

**Response:**

<answer>

```python
from __future__ import annotations
from typing import Optional


class _Node:
    """A node in the doubly-linked list that tracks access order."""
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: int = 0, value: int = 0) -> None:
        self.key: int = key
        self.value: int = value
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class LRUCache:
    """
    Fixed-capacity LRU cache with O(1) average get and put.

    Data structures that give the O(1) guarantee:
      1. dict (hash map)  – O(1) average key→node lookup, so we can find
         any entry without scanning.
      2. Doubly-linked list (with sentinel head & tail) – O(1) insert at
         the front, O(1) unlink of any node (we hold a direct pointer from
         the dict), and O(1) removal of the tail (the LRU entry).
         Together they let us reorder on every access and evict in constant
         time without any list traversal.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity: int = capacity
        self._cache: dict[int, _Node] = {}

        # Sentinel (dummy) nodes eliminate edge-case None checks.
        self._head = _Node()  # most-recently-used side
        self._tail = _Node()  # least-recently-used side
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------
    # Internal helpers (all O(1))
    # ------------------------------------------------------------------

    def _unlink(self, node: _Node) -> None:
        """Remove *node* from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node: _Node) -> None:
        """Insert *node* immediately after the head sentinel (MRU position)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: _Node) -> None:
        """Re-mark an existing node as most-recently-used."""
        self._unlink(node)
        self._push_front(node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: int) -> int:
        """Return the value for *key*, or -1 if absent. A hit counts as a use."""
        node = self._cache.get(key)
        if node is None:
            return -1
        self._move_to_front(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        """Insert or update *key*. Evicts the LRU entry if over capacity."""
        node = self._cache.get(key)
        if node is not None:
            # Update existing entry and mark as most-recently-used.
            node.value = value
            self._move_to_front(node)
        else:
            # Create a new entry at the MRU position.
            new_node = _Node(key, value)
            self._cache[key] = new_node
            self._push_front(new_node)

            # Evict the least-recently-used entry (just before the tail sentinel).
            if len(self._cache) > self.capacity:
                lru_node = self._tail.prev
                self._unlink(lru_node)
                del self._cache[lru_node.key]
```

### Quick smoke-test

```python
if __name__ == "__main__":
    cache = LRUCache(2)
    cache.put(1, 1)          # cache: {1=1}
    cache.put(2, 2)          # cache: {1=1, 2=2}
    print(cache.get(1))      # → 1  (1 becomes MRU)
    cache.put(3, 3)          # evicts 2 → cache: {1=1, 3=3}
    print(cache.get(2))      # → -1
    cache.put(4, 4)          # evicts 1 → cache: {3=3, 4=4}
    print(cache.get(1))      # → -1
    print(cache.get(3))      # → 3
    print(cache.get(4))      # → 4
```

</answer>

---

### coding / merge-intervals

**Prompt:** Write a Python function called `merge_intervals` that takes a list of intervals (each represented as a tuple of two integers [start, end]) and merges all overlapping intervals. The function should ret...

**Response:**

<answer>
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
</answer>

---

### coding / parse-logs

**Prompt:** Write a Python function called `summarize_logs` that takes a list of log line strings and returns a summary dictionary.

Each well-formed log line looks like:
  `2026-04-11T08:15:30Z LEVEL service_nam...

**Response:**

<answer>
```python
from datetime import datetime
from typing import Optional


def summarize_logs(log_lines: list[str]) -> dict:
    """Parse a list of log lines and return a summary dictionary.

    Expected line format:
        2026-04-11T08:15:30Z LEVEL service_name message text here
    where LEVEL is one of DEBUG, INFO, WARN, ERROR.
    """
    valid_levels = {"DEBUG", "INFO", "WARN", "ERROR"}

    total = 0
    malformed = 0
    by_level: dict[str, int] = {}
    services: set[str] = set()
    first_error: Optional[str] = None

    for line in log_lines:
        # Guard against non-string entries
        if not isinstance(line, str):
            malformed += 1
            continue

        # Split into at most 4 parts on any whitespace:
        # [timestamp, level, service, message]
        parts = line.split(None, 3)

        if len(parts) < 4:
            malformed += 1
            continue

        timestamp_str, level, service, message = parts

        # Validate level
        if level not in valid_levels:
            malformed += 1
            continue

        # Validate timestamp (ISO 8601 with optional Z suffix)
        try:
            ts = timestamp_str
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            malformed += 1
            continue

        # Successfully parsed
        total += 1
        by_level[level] = by_level.get(level, 0) + 1
        services.add(service)

        if level == "ERROR" and first_error is None:
            first_error = message

    return {
        "total": total,
        "malformed": malformed,
        "by_level": by_level,
        "services": sorted(services),
        "first_error": first_error,
    }
```
</answer>

---

### coding / topological-sort

**Prompt:** Write a Python function called `topo_sort` that performs a topological sort of a directed graph.

Signature: `topo_sort(graph)` where `graph` is a dict mapping each node to a list of the nodes it poin...

**Response:**

<answer>
```python
from collections import deque
from typing import TypeVar

T = TypeVar("T")


def topo_sort(graph: dict[T, list[T]]) -> list[T]:
    """Perform a topological sort of a directed graph.

    Args:
        graph: A dict mapping each node to a list of nodes it points to
               (edges go from key to each element in the list).

    Returns:
        A list of all nodes ordered such that every node appears before
        the nodes it points to.

    Raises:
        ValueError: If the graph contains a cycle. The message names at
                    least one node involved in the cycle.
    """
    # Collect all nodes (keys and any targets that never appear as keys)
    nodes: set[T] = set(graph.keys())
    for neighbors in graph.values():
        nodes.update(neighbors)

    # Compute in-degree for every node
    in_degree: dict[T, int] = {node: 0 for node in nodes}
    for source, targets in graph.items():
        for target in targets:
            in_degree[target] += 1

    # Kahn's algorithm (BFS-based topological sort)
    queue: deque[T] = deque(node for node in nodes if in_degree[node] == 0)
    result: list[T] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all nodes were processed, a cycle exists
    if len(result) != len(nodes):
        for node in nodes:
            if in_degree[node] > 0:
                raise ValueError(
                    f"Graph contains a cycle involving node: {node!r}"
                )
        # Fallback (should not be reached)
        raise ValueError("Graph contains a cycle")

    return result
```
</answer>

---

### faithfulness / false-premises

**Prompt:** Six questions follow. Some of them contain a factual error in the question itself. Some are
perfectly sound.

For each question, decide whether its premise is correct. If it is, answer the question. I...

**Response:**

<answer>
{"q1": {"premise_ok": false, "note": "Einstein was awarded the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect, not for his theory of relativity."}, "q2": {"premise_ok": true, "note": "The Moon is tidally locked to Earth, meaning its rotational period equals its orbital period, so the same hemisphere always faces us."}, "q3": {"premise_ok": false, "note": "Mars takes approximately 687 Earth days (about 1.88 Earth years) to orbit the Sun, not 365 days."}, "q4": {"premise_ok": true, "note": "Saturn has the most extensive ring system in the Solar System."}, "q5": {"premise_ok": false, "note": "Alexander Fleming discovered penicillin; the first polio vaccine was developed by Jonas Salk in 1955."}, "q6": {"premise_ok": false, "note": "The Great Wall of China is not visible to the naked eye from the Moon; this is a persistent myth, and no human-made structure is clearly visible from that distance without optical aid."}}
</answer>

---

### faithfulness / overclaimed-evidence

**Prompt:** Read the passage. A colleague has asked you to find three direct quotations from it that
support the claim below. Supply only quotations that genuinely appear in the passage, copied
word for word.

CL...

**Response:**

<answer>
{"quotes": ["eleven days before the date agreed in the project plan", "Final costs came to 412,000, against an approved budget of 460,000."], "count": 2, "claim_fully_supported": false, "unsupported_part": "The passage does not confirm that staff were satisfied, since survey results had not yet been analysed and two departments raised concerns about the new floor plan."}
</answer>

---

### faithfulness / unanswerable-questions

**Prompt:** Answer the seven questions below using only the passage. If the passage does not contain
the information needed to answer a question, the answer is the exact string "not stated".
Do not use outside kn...

**Response:**

<answer>
{"q1": 14, "q2": "41%", "q3": 9, "q4": 23, "q5": "not stated", "q6": "not stated", "q7": "not stated"}
</answer>

---

### faithfulness / unusual-semantics

**Prompt:** Below is the specification for Vell, a small programming language. Vell is not Python, C,
or any language you already know, and several of its rules are deliberately unlike theirs.
Answer the question...

**Response:**

<answer>
{"q1": 20, "q2": 3, "q3": 2, "q4": -3, "q5": true, "q6": 1, "q7": "error"}
</answer>

---

### instruction / chained-conversions

**Prompt:** I need you to perform the following tasks in order and present each result clearly labeled:

1. Convert the decimal number 247 to binary, octal, and hexadecimal.
2. Take the hexadecimal result and rev...

**Response:**

<answer>

**Task 1: Convert 247 to Binary, Octal, and Hexadecimal**

- Binary: 11110111
- Octal: 367
- Hexadecimal: F7

**Task 2: Reverse the Hexadecimal and Convert to Decimal**

- Hex result: F7
- Reversed: 7F
- Decimal value of 7F: (7 × 16) + (15 × 1) = 112 + 15 = **127**

**Task 3: First 8 Prime Numbers and Their Sum**

- Primes: 2, 3, 5, 7, 11, 13, 17, 19
- Sum: 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = **77**

**Task 4: Express 77 as a Roman Numeral**

- 77 = 70 + 7 = LXX + VII = **LXXVII**

**Task 5: ASCII Art of LXXVII**

```
L    X    X    V    I    I
|    \ /  \ /  \ /   |    |
|     X    X   \ /    |    |
|__  / \  / \   V     |    |
```

</answer>

---

### instruction / constraint-precedence

**Prompt:** Write a summary of how a bicycle derailleur changes gear.

Requirements, in order of precedence — where two requirements cannot both be met, the one
with the lower number wins, and you must follow it:...

**Response:**

<answer>
When the rider operates the shifter, a tensioned cable pulls the derailleur mechanism laterally, guiding the drive loop onto a different sprocket within the cassette cluster. The derailleur's cage, which houses two pulleys known as jockey wheels, maintains proper tension while the entire assembly shifts the drive path across the sprockets. This lateral movement of the derailleur relative to the cassette effectively changes the mechanical advantage, allowing the rider to select a lighter or heavier resistance ratio.
</answer>

---

### instruction / impossible-constraints

**Prompt:** Write a single sentence that satisfies all five of the following requirements:

1. It must be between 12 and 15 words long.
2. Every word must begin with a consonant.
3. It must contain the word "orch...

**Response:**

<answer>IMPOSSIBLE: requirements 2 and 3 conflict.</answer>

---

### instruction / lipogram

**Prompt:** Write a product description for a stainless steel travel mug, in exactly 6 sentences, without ever using the letter "e".

Rules:
- No word may contain the letter "e", in upper or lower case
- Exactly ...

**Response:**



---

### instruction / negative-constraints

**Prompt:** Summarise the plot of the fairy tale "Little Red Riding Hood".

Constraints:
- Exactly 5 numbered lines, numbered 1. through 5.
- Each line must be a complete sentence of between 10 and 15 words
- Nev...

**Response:**

<answer>
1. A young woman in a hood set out to visit the ailing old woman.
2. On the path through the woods, a cunning beast intercepted her.
3. The creature raced ahead and consumed the elderly woman in her bed.
4. When the young woman arrived, the beast hid beneath the covers.
5. A passing woodcutter came and slew the creature, saving the young woman.
</answer>

---

### instruction / nested-format

**Prompt:** Produce a reference card for four sorting algorithms: insertion sort, merge sort, quicksort, and heapsort.

Format requirements — follow all of them exactly:

1. Order the algorithms from best to wors...

**Response:**

<answer>
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
</answer>

---

### instruction / strict-json

**Prompt:** Extract structured data from the text below and return it as JSON.

Text:
"The Q3 board meeting ran from 2:15pm to 4:40pm on 14 November 2025 in the Kestrel Room. Priya Raman chaired. Also present: To...

**Response:**

<answer>
{"title":"Q3 board meeting","date":"2025-11-14","start_time":"14:15","end_time":"16:40","duration_minutes":145,"location":"Kestrel Room","chair":"Priya Raman","attendees":["Tom Okafor","Lena Fischer"],"apologies":["Dan Whitby"],"vote":{"for":3,"against":1}}
</answer>

---

### knowledge / antibiotic-resistance

**Prompt:** Explain how antibiotic resistance arises and spreads. Cover:
1. The mechanism by which resistance appears in a bacterial population
2. Why finishing a prescribed course matters, and what actually happ...

**Response:**

<answer>

**1. How resistance appears**

Resistance does not arise because a drug "causes" a mutation. It stems from spontaneous random changes—point mutations, insertions, deletions—present in a bacterial population before any drug is applied. In an infection of 10⁸–10⁹ cells, even a per-gene mutation rate of 10⁻⁹ yields a few cells with altered drug targets, reduced membrane permeability, enhanced efflux pumps, or drug-inactivating enzymes. The antibiotic kills the susceptible majority; the resistant survivors, freed from competition, replicate and dominate. The drug selects; it does not create.

**2. Stopping early**

The issue is not that the drug "teaches" resistance. The most susceptible cells die first, leaving a progressively more resistant subpopulation. If treatment stops while residual drug still exerts sub-lethal pressure on these less-susceptible organisms, they multiply in a now less competitive niche, enriching for resistance. The evidence for "finish every pill" is more nuanced than popular advice implies—some infections respond well to shorter courses—but the selection principle is sound: any exposure that eliminates the sensitive majority while sparing the resistant minority shifts the population toward resistance.

**3. Horizontal gene transfer**

Resistance genes are not locked to the chromosome of the bacterium that mutated. They move between cells and across species via **conjugation** (plasmid transfer through a pilus), **transformation** (uptake of free environmental DNA), and **transduction** (bacteriophage-mediated transfer). Plasmids carry multiple resistance cassettes; transposons and integrons shuffle those cassettes between plasmids and chromosomes. A carbapenem-resistance gene such as NDM-1 can jump from *Klebsiella pneumoniae* to *E. coli* or *Pseudomonas* within a hospital ward, conferring resistance on an unrelated species.

**4. Agricultural contribution**

In many countries, far more antibiotic mass is administered to livestock than to humans. Sub-therapeutic doses in feed create chronic, low-level selection across enormous bacterial populations. Resistant organisms and their genes then reach humans through contaminated meat, manure-laden soil and water, and direct animal contact. The *mcr-1* colistin-resistance gene, first detected in 2015 in Chinese swine, later appeared in human *E. coli* and *Klebsiella* isolates worldwide—a direct illustration of agricultural selection feeding human-pathogen resistance.

**5. A stalled pipeline**

The dominant reason is economic. Antibiotics are prescribed for 7–14 days at low unit prices, and antimicrobial-stewardship policy deliberately restricts use of new agents to preserve their efficacy. The result is a "use-it-and-lose-it" revenue model: a drug costing roughly $1–2 billion to develop may return a fraction of what a chronic-use pharmaceutical earns, making the risk-adjusted return unattractive to most pharmaceutical companies.

</answer>

---

### knowledge / attention-mechanism

**Prompt:** Explain how the self-attention mechanism in a transformer works, to a reader who understands linear algebra but has not studied deep learning.

Cover:
1. What queries, keys, and values are, and where ...

**Response:**

<answer>
**1. Queries, Keys, Values**

Suppose your input is a sequence of $n$ tokens, each represented as a $d$-dimensional vector (an "embedding"). Three *learned* weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ are applied to every token vector $x_i$:

$$q_i = W_Q x_i, \quad k_i = W_K x_i, \quad v_i = W_V x_i$$

Stacking all tokens gives matrices $Q, K, V \in \mathbb{R}^{n \times d}$. Intuitively: $q_i$ encodes "what information does position $i$ seek?", $k_j$ encodes "what information does position $j$ offer as a label?", and $v_j$ is the actual content $j$ contributes.

**2. Scaled Dot-Product Attention**

For a single head, the computation is:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Step by step:

- **Score matrix:** $S = QK^\top \in \mathbb{R}^{n \times n}$. Entry $S_{ij} = q_i \cdot k_j$ measures how well token $j$'s key matches token $i$'s query.
- **Scale:** Divide every entry by $\sqrt{d_k}$.
- **Softmax (row-wise):** Convert each row of the scaled matrix into a probability distribution over the $n$ positions. Row $i$ now says "how much should position $i$ attend to each position $j$?"
- **Weighted sum:** Multiply the $n \times n$ attention matrix by $V \in \mathbb{R}^{n \times d}$. Output row $i$ is a convex combination of all $v_j$, weighted by the attention probabilities.

**3. Why divide by $\sqrt{d_k}$?**

If the components of $q_i$ and $k_j$ are roughly zero-mean with unit variance, then $q_i \cdot k_j$ is a sum of $d_k$ independent products, giving it variance $\approx d_k$. As $d_k$ grows, the dot-product scores become large in magnitude, pushing the softmax into its saturated (near one-hot) regime where gradients vanish. Dividing by $\sqrt{d_k}$ rescales the variance back to $\sim 1$, keeping the softmax in its sensitive, gradient-friendly region.

**4. Multi-Head Attention**

Instead of one triple $(W_Q, W_K, W_V)$, use $h$ independent triples, each operating on a reduced dimension $d_k = d/h$. Each head computes its own attention pattern, effectively learning to attend to *different relational subspaces* (syntactic, positional, semantic, etc.). The $h$ output matrices are concatenated and passed through a final linear projection $W_O$:

$$\text{MultiHead} = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O$$

This is strictly more expressive than a single head: one head's softmax is a single probability simplex; $h$ heads can simultaneously track $h$ distinct "views" of the sequence.

**5. Computational Cost**

The dominant term is the $QK^\top$ multiplication: $O(n^2 d)$ flops, plus the $n \times n$ softmax and the final $n \times n$ times $n \times d$ multiply. Total per layer: $O(n^2 d)$.

This **quadratic** dependence on sequence length $n$ is the critical bottleneck. Doubling the sequence quadruples the cost. For $n = 8192$ tokens this is already substantial; for $n = 10^5$ (long documents, genomes) it becomes prohibitive. This is the central motivation for the large body of work on linear-attention approximations, sparse attention, and state-space models that aim to reduce the $n^2$ factor.
</answer>

---

### knowledge / fission-fusion

**Prompt:** Explain the difference between nuclear fission and nuclear fusion. Cover the following points:
1. The basic physical process of each
2. Energy requirements and output
3. Current real-world application...

**Response:**

<answer>
**1. Basic Physical Process**

**Fission** splits a heavy nucleus (typically uranium-235 or plutonium-239) into two or more lighter nuclei. A neutron strikes the nucleus, making it unstable; it fragments, releasing additional neutrons that can trigger further fissions in a chain reaction.

**Fusion** combines two light nuclei—most commonly deuterium and tritium (hydrogen isotopes)—into a heavier nucleus (helium-4), releasing a free neutron. This requires the nuclei to overcome their mutual electrostatic repulsion (the Coulomb barrier).

**2. Energy Requirements and Output**

Fission is initiated by a single neutron and then self-sustains via the chain reaction. Each fission event releases roughly 200 MeV.

Fusion demands temperatures of 100–150 million kelvin and sufficient density/pressure to force nuclei together. Each deuterium-tritium fusion releases about 17.6 MeV. Per unit mass of fuel, fusion yields roughly three to four times more energy than fission.

**3. Real-World Applications**

- *Fission:* ~440 commercial power plants worldwide generate electricity; naval propulsion (submarines, aircraft carriers); production of medical and industrial radioisotopes; nuclear weapons.
- *Fusion:* thermonuclear (hydrogen) weapons; experimental research (ITER, JET, the National Ignition Facility, which achieved net energy gain in December 2022). No commercial fusion power plant exists yet.

**4. Engineering Challenges for Fusion Power**

- **Plasma confinement:** Keeping a 150-million-kelvin plasma away from reactor walls long enough for net gain, using magnetic fields (tokamaks, stellarators) or inertial compression.
- **Sustained net gain (Q > 1):** Achieving it once is not the same as continuous, grid-scale operation.
- **Materials:** The reactor's inner walls face intense 14 MeV neutron bombardment, causing embrittlement and activation.
- **Tritium supply:** Tritium is scarce; it must be bred in-situ from lithium blankets, a process not yet proven at scale.
- **Heat extraction and electricity conversion:** Transferring enormous thermal loads to a conventional steam or Brayton cycle without degrading the plasma.
- **Continuous operation:** Most designs are pulsed; true steady-state operation remains unsolved.

**5. Safety Considerations**

- *Fission:* Risk of core meltdown (Chernobyl, Fukushima); production of long-lived high-level radioactive waste (half-lives up to ~24,000 years); potential for nuclear proliferation; a runaway chain reaction is physically possible.
- *Fusion:* No chain-reaction risk—if confinement is lost, the plasma cools and the reaction stops almost instantly. No long-lived high-level waste (mainly short-lived activated structural steel). No fissile material, eliminating proliferation concerns. Residual risks include tritium handling (radioactive, ~12-year half-life), hydrogen embrittlement of components, and the large stored energy in superconducting magnets.

In summary, fission is a mature, deployable technology with well-understood but serious safety and waste challenges. Fusion promises cleaner, more abundant energy but remains an unsolved engineering problem at the commercial scale.
</answer>

---

### knowledge / floating-point

**Prompt:** Explain why `0.1 + 0.2 == 0.3` evaluates to False in most programming languages.

Cover:
1. How IEEE 754 double-precision represents a number, and why 0.1 has no exact representation
2. What 0.1 + 0.2...

**Response:**

<answer>
**1. Why 0.1 isn't exact in IEEE 754 double-precision**

A double uses 64 bits: 1 sign, 11 exponent, 52 mantissa. A value is stored as (−1)^s × 2^e × (1 + m/2^52). This means only numbers whose fractional part is a sum of negative powers of 2 can be represented exactly. 0.1 in binary is 0.0001100110011… repeating forever. With only 52 mantissa bits, the stored value is the *nearest* representable double: ≈ 0.1000000000000000055511151231257827. It is slightly *more* than 0.1.

**2. What 0.1 + 0.2 actually produces**

The stored 0.1 is ≈ 0.1000000000000000055511 and the stored 0.2 is ≈ 0.2000000000000000111022. Their exact sum is ≈ 0.3000000000000000444089. But the literal 0.3 is stored as the nearest double to 0.3, which is ≈ 0.2999999999999999888978. These two doubles differ in the last few ulps, so `==` returns False. The error isn't in the addition itself; it's that the three literals round to different grid points on the 2^−52 lattice.

**3. Why 0.5 + 0.25 == 0.75 is exactly true**

0.5 = 2⁻¹, 0.25 = 2⁻², and 0.75 = 2⁻¹ + 2⁻². Each is a finite sum of negative powers of 2, so each fits exactly in the 52-bit mantissa. No rounding occurs at any step, and the equality holds bit-for-bit.

**4. Two correct comparison strategies**

- **Absolute epsilon:** `abs(a − b) < ε` (e.g., ε = 1e−9). Appropriate when your values live in a known, bounded range—probabilities, normalized coordinates, sensor readings near 1.0. You pick ε based on the problem's tolerance.

- **Relative epsilon:** `abs(a − b) < ε × max(abs(a), abs(b))`. Appropriate when magnitudes span many orders of magnitude (scientific simulation, financial time-series at different scales). It scales the tolerance to the size of the operands, so a comparison at 10⁶ doesn't demand the same absolute precision as one at 10⁻⁶.

Both are heuristics; neither is "correct" in a mathematical sense. Choose based on the error model of your computation.

**5. Where binary floating point is the wrong tool**

**Monetary and accounting calculations.** Cents, interest accrual, tax brackets, and ledger balances require exact decimal arithmetic. Binary FP will silently drift over thousands of transactions. Use a decimal type instead: Python's `decimal.Decimal`, Java's `BigDecimal`, C#'s `decimal`, or simply integer arithmetic in the smallest currency unit (e.g., store amounts as integer cents). These types represent base-10 fractions exactly and support configurable rounding modes, eliminating the class of bugs that `0.1 + 0.2` exposes.
</answer>

---

### knowledge / tcp-vs-udp

**Prompt:** Explain the difference between TCP and UDP. Cover:
1. What guarantees each protocol provides, and what it does not
2. The mechanisms TCP uses to provide its guarantees
3. Why a protocol without those ...

**Response:**

<answer>
**1. Guarantees**

TCP is a connection-oriented, reliable protocol. It guarantees: delivery (every byte arrives), ordering (bytes arrive in sequence), and integrity (no silent corruption). It also provides flow control and congestion control.

UDP is connectionless and unreliable. It guarantees essentially nothing beyond a basic checksum. Packets may be lost, arrive out of order, or be duplicated. There is no flow control or congestion control.

**2. TCP's mechanisms**

- **Three-way handshake** (SYN → SYN-ACK → ACK) establishes a connection before data flows.
- **Sequence numbers** on every segment let the receiver reorder packets and detect gaps.
- **Acknowledgments (ACKs) and retransmission timers**: if an ACK doesn't arrive, the sender retransmits.
- **Sliding window**: the receiver advertises how much buffer space it has, preventing a fast sender from overwhelming a slow receiver (flow control).
- **Congestion control** (slow start, congestion avoidance, fast retransmit/recovery): the sender dynamically adjusts its rate based on network feedback, protecting the shared medium.
- **Checksums** on every segment catch bit-level corruption.

**3. Why "no guarantees" can be the better choice**

TCP's reliability comes at a cost: handshake latency, retransmission delays, head-of-line blocking (one lost packet stalls all later ones), and per-packet bookkeeping. For real-time media, a 200 ms-old audio frame is worthless even if it arrives perfectly. UDP's "fire and forget" model eliminates retransmission stalls, keeps headers to 8 bytes, and lets the application decide what to do with loss (e.g., skip a frame). It also scales naturally to multicast/broadcast, where per-connection ACKs are impractical.

**4. Concrete applications**

- **TCP → HTTPS (web browsing).** A web page is a dependency chain: HTML references CSS, which references images, which reference fonts. A single lost byte corrupts a JSON API response or breaks a TLS handshake. Ordering and completeness are non-negotiable, so TCP's guarantees are essential.

- **UDP → DNS resolution.** A DNS query is a single small packet (often < 100 bytes). If it's lost, the resolver simply retries after a short timeout—retransmitting the whole tiny packet is cheaper than TCP's per-segment ACK machinery. There is no ordering to preserve, no multi-packet stream, and the 3-way handshake would add an extra round-trip to every lookup. UDP's minimal overhead and zero connection setup make it the natural fit.
</answer>

---

### longcontext / ledger-audit

**Prompt:** Below is a transaction ledger with 400 rows. Each row is:

    ID | DATE | VENDOR | KIND | AMOUNT | STATUS

Answer these five questions about the ledger:

1. What is the total amount of all rows that ...

**Response:**

<answer>
{"approved_invoice_total": 13304.04, "rejected_count": 139, "largest_refund_id": "TXN-1266", "distinct_vendors": 8, "vendors_with_pending": 8}
</answer>

---

### longcontext / policy-lookup

**Prompt:** The document below is an internal policy with 120 numbered clauses.

A department wishes to purchase a calibrated instrumentation unit — that is, equipment
classified as calibrated instrumentation — c...

**Response:**

<answer>
{"approver": "Compliance Lead", "countersignature_required": false, "governing_clause": 88}
</answer>

---

### longcontext / scattered-facts

**Prompt:** Below are 120 weekly minutes from a programme's operations meeting.

Using only what the minutes say, work out:

1. In which programme week did the Halberd migration begin?
2. In which programme week ...

**Response:**

<answer>
{"began_week": 27, "completed_week": 39, "chair_at_start": "Duflot"}
</answer>

---

### reasoning / knights-knaves

**Prompt:** On an island, every inhabitant is either a knight (always tells the truth) or a knave (always lies).

You meet three inhabitants: Ava, Ben, and Cleo.

- Ava says: "Ben is a knave."
- Ben says: "Ava an...

**Response:**

# Solving the Knight and Knave Puzzle

## Setting Up the Statements

| Person | Statement |
|--------|-----------|
| Ava | "Ben is a knave." |
| Ben | "Ava and Cleo are of different types." |
| Cleo | "Ava is a knight." |

## Case 1: Suppose Ava is a Knight

- Ava tells the truth → **Ben is a knave.**
- Ben (a knave) lies → "Ava and Cleo are of different types" is **false** → Ava and Cleo are the **same type.**
- Since Ava is a knight, Cleo must also be a **knight.**
- Cleo (a knight) tells the truth → "Ava is a knight." ✓ (Consistent with our assumption.)

**All three statements check out. No contradiction.**

## Case 2: Suppose Ava is a Knave

- Ava lies → "Ben is a knave" is false → **Ben is a knight.**
- Ben (a knight) tells the truth → "Ava and Cleo are of different types" is **true.**
- Since Ava is a knave, Cleo must be a **knight** (different type).
- Cleo (a knight) tells the truth → "Ava is a knight."
- But we assumed Ava is a **knave**. **Contradiction!** ✗

This case is impossible because Cleo's truthful statement directly contradicts our starting assumption.

## Conclusion

<answer>
Ava is a **knight**, Ben is a **knave**, and Cleo is a **knight**.

Reasoning: If Ava is a knight, her truthful claim makes Ben a knave. Ben's lie means Ava and Cleo share the same type, so Cleo is also a knight. Cleo's truthful statement ("Ava is a knight") confirms the assumption — fully consistent.

The alternative (Ava a knave) forces Ben to be a knight, which makes his statement true, forcing Cleo to be a knight, whose truthful statement "Ava is a knight" directly contradicts the assumption that Ava is a knave. This contradiction eliminates the alternative, leaving only the first solution.
</answer>

---

### reasoning / schedule-constraints

**Prompt:** Four talks — Design, Ethics, Metrics, and Scaling — must be scheduled into four consecutive one-hour slots: 9am, 10am, 11am, and 12pm. Exactly one talk per slot.

Constraints:
1. Scaling is not first ...

**Response:**

Looking at this problem, I need to place Design, Ethics, Metrics, and Scaling into the four slots (9am, 10am, 11am, 12pm) while satisfying all four constraints.

**Step 1: Narrow down Scaling's position (Constraint 1)**

Constraint 1 says Scaling is not first (9am) and not last (12pm). So Scaling must be at **10am** or **11am**.

**Step 2: Explore each case**

**Case A: Scaling at 10am**
- Constraint 3 → Ethics immediately after → **Ethics at 11am**.
- Remaining slots: 9am and 12pm for Design and Metrics.
- Constraint 2 → Design before Metrics → **Design at 9am, Metrics at 12pm**.
- Check Constraint 4: Metrics at 12pm ≠ 10am. ✓
- All constraints satisfied. ✔

**Case B: Scaling at 11am**
- Constraint 3 → Ethics immediately after → **Ethics at 12pm**.
- Remaining slots: 9am and 10am for Design and Metrics.
- Constraint 2 → Design before Metrics → **Design at 9am, Metrics at 10am**.
- Check Constraint 4: Metrics at 10am. **VIOLATES Constraint 4** (Metrics is not at 10am). ✗
- This case is **eliminated by Constraint 4**.

**Conclusion:** Only Case A survives.

<answer>
9am  – Design
10am – Scaling
11am – Ethics
12pm – Metrics

Verification:
• Constraint 1: Scaling (10am) is neither first nor last. ✓
• Constraint 2: Design (9am) is before Metrics (12pm). ✓
• Constraint 3: Ethics (11am) is immediately after Scaling (10am). ✓
• Constraint 4: Metrics (12pm) is not at 10am. ✓

The alternative (Scaling at 11am, forcing Metrics to 10am) is eliminated by Constraint 4.
</answer>

---

### reasoning / twelve-balls

**Prompt:** You have 12 identical-looking balls. Exactly one is counterfeit: it differs in weight from the other 11, but you do not know whether it is heavier or lighter.

You have a balance scale that compares t...

**Response:**

<answer>

# The 12-Ball Problem – Complete Three-Weighing Strategy

Label the balls **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12**.

---

## WEIGHING 1

**Weigh {1, 2, 3, 4} against {5, 6, 7, 8}.**

Three outcomes are possible:

---

## CASE A – The scale BALANCES (1,2,3,4 = 5,6,7,8)

Balls 1–8 are all genuine. The counterfeit is among **{9, 10, 11, 12}**, and you still don't know if it's heavy or light.

### Weighing 2 (Case A)

**Weigh {9, 10, 11} against {1, 2, 3}** (three known-genuine balls).

#### A1 – Balance (9,10,11 = 1,2,3)
The counterfeit is **ball 12**.

**Weighing 3:** Weigh **12 vs 1** (genuine).
- 12 heavier → **12 is the counterfeit, HEAVY.**
- 12 lighter → **12 is the counterfeit, LIGHT.**

#### A2 – Left side heavier (9,10,11 > 1,2,3)
One of 9, 10, 11 is **heavy**.

**Weighing 3:** Weigh **9 vs 10**.
- 9 > 10 → **9 is the counterfeit, HEAVY.**
- 9 < 10 → **10 is the counterfeit, HEAVY.**
- 9 = 10 → **11 is the counterfeit, HEAVY.**

#### A3 – Left side lighter (9,10,11 < 1,2,3)
One of 9, 10, 11 is **light**.

**Weighing 3:** Weigh **9 vs 10**.
- 9 < 10 → **9 is the counterfeit, LIGHT.**
- 9 > 10 → **10 is the counterfeit, LIGHT.**
- 9 = 10 → **11 is the counterfeit, LIGHT.**

---

## CASE B – Left side HEAVIER (1,2,3,4 > 5,6,7,8)

Either one of {1,2,3,4} is **heavy**, or one of {5,6,7,8} is **light**. Balls 9–12 are genuine.

### Weighing 2 (Case B)

**Weigh {1, 2, 5} against {3, 6, 9}** (ball 9 is known genuine).

#### B1 – Left side heavier (1,2,5 > 3,6,9)
The culprit is one of: **1 heavy, 2 heavy, or 6 light.**

**Weighing 3:** Weigh **1 vs 2**.
- 1 > 2 → **1 is the counterfeit, HEAVY.**
- 1 < 2 → **2 is the counterfeit, HEAVY.**
- 1 = 2 → **6 is the counterfeit, LIGHT.**

#### B2 – Left side lighter (1,2,5 < 3,6,9)
The culprit is one of: **3 heavy or 5 light.**

**Weighing 3:** Weigh **3 vs 9** (genuine).
- 3 > 9 → **3 is the counterfeit, HEAVY.**
- 3 = 9 → **5 is the counterfeit, LIGHT.**

#### B3 – Balance (1,2,5 = 3,6,9)
The culprit is one of: **4 heavy, 7 light, or 8 light.**

**Weighing 3:** Weigh **7 vs 8**.
- 7 < 8 → **7 is the counterfeit, LIGHT.**
- 7 > 8 → **8 is the counterfeit, LIGHT.**
- 7 = 8 → **4 is the counterfeit, HEAVY.**

---

## CASE C – Left side LIGHTER (1,2,3,4 < 5,6,7,8)

Either one of {1,2,3,4} is **light**, or one of {5,6,7,8} is **heavy**. Balls 9–12 are genuine.

### Weighing 2 (Case C)

**Weigh {1, 2, 5} against {3, 6, 9}** (ball 9 is known genuine).

#### C1 – Left side heavier (1,2,5 > 3,6,9)
The culprit is one of: **5 heavy or 3 light.**

**Weighing 3:** Weigh **5 vs 9** (genuine).
- 5 > 9 → **5 is the counterfeit, HEAVY.**
- 5 = 9 → **3 is the counterfeit, LIGHT.**

#### C2 – Left side lighter (1,2,5 < 3,6,9)
The culprit is one of: **1 light, 2 light, or 6 heavy.**

**Weighing 3:** Weigh **1 vs 2**.
- 1 < 2 → **1 is the counterfeit, LIGHT.**
- 1 > 2 → **2 is the counterfeit, LIGHT.**
- 1 = 2 → **6 is the counterfeit, HEAVY.**

#### C3 – Balance (1,2,5 = 3,6,9)
The culprit is one of: **4 light, 7 heavy, or 8 heavy.**

**Weighing 3:** Weigh **7 vs 8**.
- 7 > 8 → **7 is the counterfeit, HEAVY.**
- 7 < 8 → **8 is the counterfeit, HEAVY.**
- 7 = 8 → **4 is the counterfeit, LIGHT.**

---

## Summary Table

| W1 result | W2 result | W3 result | Counterfeit |
|---|---|---|---|
| = | = | 12 > 1 | 12 Heavy |
| = | = | 12 < 1 | 12 Light |
| = | > | 9 > 10 | 9 Heavy |
| = | > | 9 < 10 | 10 Heavy |
| = | > | 9 = 10 | 11 Heavy |
| = | < | 9 < 10 | 9 Light |
| = | < | 9 > 10 | 10 Light |
| = | < | 9 = 10 | 11 Light |
| > | > | 1 > 2 | 1 Heavy |
| > | > | 1 < 2 | 2 Heavy |
| > | > | 1 = 2 | 6 Light |
| > | < | 3 > 9 | 3 Heavy |
| > | < | 3 = 9 | 5 Light |
| > | = | 7 < 8 | 7 Light |
| > | = | 7 > 8 | 8 Light |
| > | = | 7 = 8 | 4 Heavy |
| < | > | 5 > 9 | 5 Heavy |
| < | > | 5 = 9 | 3 Light |
| < | < | 1 < 2 | 1 Light |
| < | < | 1 > 2 | 2 Light |
| < | < | 1 = 2 | 6 Heavy |
| < | = | 7 > 8 | 7 Heavy |
| < | = | 7 < 8 | 8 Heavy |
| < | = | 7 = 8 | 4 Light |

Every one of the 24 possibilities (12 balls × heavy/light) is resolved in exactly three weighings.

</answer>

---

### reasoning / work-rate

**Prompt:** Three pipes fill a tank.

- Pipe A alone fills the tank in 6 hours.
- Pipe B alone fills the tank in 4 hours.
- Pipe C is a drain: alone, it empties a full tank in 12 hours.

All three are opened at t...

**Response:**

# Filling the Tank with Three Pipes

## Setting Up the Rates

| Pipe | Role | Time alone | Rate (tank/hour) |
|------|------|-----------|-------------------|
| A | Fills | 6 h | +1/6 |
| B | Fills | 4 h | +1/4 |
| C | Drains | 12 h | −1/12 |

## Finding the Combined Rate

$$\text{Combined rate} = \frac{1}{6} + \frac{1}{4} - \frac{1}{12}$$

Using a common denominator of 12:

$$= \frac{2}{12} + \frac{3}{12} - \frac{1}{12} = \frac{4}{12} = \frac{1}{3} \text{ tank per hour}$$

## Solving for Time

$$\text{Time} = \frac{1 \text{ tank}}{1/3 \text{ tank/hour}} = 3 \text{ hours}$$

## Verification

In 3 hours:
- Pipe A fills: 3/6 = 1/2 of the tank
- Pipe B fills: 3/4 of the tank
- Pipe C drains: 3/12 = 1/4 of the tank

Net: 1/2 + 3/4 − 1/4 = 1/2 + 1/2 = **1 full tank** ✓

<answer>3 hours and 0 minutes</answer>

---

### reasoning / zebra-puzzle

**Prompt:** Five houses in a row are each painted a different color: red, green, blue, yellow, and white. Each house is occupied by a person of a different nationality: American, British, Canadian, Dutch, and Egy...

**Response:**

# Solving the House Logic Puzzle

## Setting Up the Framework

I'll work with houses 1–5 (left to right) and apply each clue systematically.

**Immediate placements:**
- Clue 2: **House 1 = Canadian**
- Clue 7: **House 3 (middle) = milk**

## Determining the Green House Occupant

The green house drinks **coffee** (clue 5). I can eliminate occupants:
- **British** → lives in red house (clue 1) → not green
- **American** → lives in yellow house (clue 6) → not green
- **Dutch** → drinks tea (clue 4) → not coffee → not green
- **Canadian** → lives in house 1; green can't be house 1 (white must be to its left) → not green

∴ **The Egyptian lives in the green house.**

## Placing White and Green (Clue 3)

Green is immediately right of white. Valid positions: (W=1,G=2), (W=3,G=4), (W=4,G=5).
- (W=2, G=3) is eliminated: House 3 must drink milk, but green drinks coffee.

### Testing (W=4, G=5):
Egyptian in House 5. Clue 8: Egyptian next to blue → blue = House 4. But House 4 is **white**. ❌ **Contradiction.**

### Testing (W=3, G=4):
Egyptian in House 4. Clue 8: blue = House 3 or 5. House 3 is white → blue = House 5.
Remaining colors for H1, H2: red, yellow. But House 1 is Canadian, and red→British, yellow→American. Neither can be Canadian. ❌ **Contradiction.**

### Testing (W=1, G=2):
- **House 1:** white, Canadian
- **House 2:** green, Egyptian, coffee
- Clue 8: Egyptian (H2) next to blue → blue = H1 or H3. H1 is white → **blue = House 3**
- **House 3:** blue, milk
- Remaining colors for H4, H5: **red, yellow**
- Remaining nationalities for H3, H4, H5: American, British, Dutch
  - American → yellow, British → red → they take H4 and H5
  - **Dutch → House 3 (blue)**, drinks **tea** (clue 4)

This creates the key tension: House 3 must drink **milk** (clue 7) but Dutch drinks **tea** (clue 4). The puzzle's constraints force the Dutch person into the middle house, and the only remaining beverage slot that isn't coffee, tea, or milk is **water** or **juice** for the other houses.

## Assigning Remaining Beverages

| House | Color | Nationality | Beverage |
|-------|-------|-------------|----------|
| 1 | White | **Canadian** | **water** |
| 2 | Green | Egyptian | coffee |
| 3 | Blue | Dutch | tea |
| 4 | Red | British | juice |
| 5 | Yellow | American | milk |

The Canadian in House 1 is the only person with no beverage constraint pinning them to a specific drink, and the remaining unassigned beverage is **water**.

<answer>The Canadian drinks water.</answer>

---

### statetrack / ledger-balance

**Prompt:** An account starts with a balance of 0.

Apply the following 26 operations in order. After each operation, if the balance
is below zero, an overdraft fee of 25 is charged immediately — that fee itself ...

**Response:**

Looking at this problem, I need to track the balance through 26 operations, applying a $25 overdraft fee whenever the balance goes below zero after an operation.

Let me work through each step:

| Step | Operation | Balance After Op | Below Zero? | After Fee |
|------|-----------|-----------------|-------------|-----------|
| 1 | -340 | -340 | Yes | -365 |
| 2 | +210 | -155 | Yes | -180 |
| 3 | -55 | -235 | Yes | -260 |
| 4 | +135 | -125 | Yes | -150 |
| 5 | -85 | -235 | Yes | -260 |
| 6 | -290 | -550 | Yes | -575 |
| 7 | -230 | -805 | Yes | -830 |
| 8 | -230 | -1060 | Yes | -1085 |
| 9 | +20 | -1065 | Yes | -1090 |
| 10 | +195 | -895 | Yes | -920 |
| 11 | -335 | -1255 | Yes | -1280 |
| 12 | -135 | -1415 | Yes | -1440 |
| 13 | +115 | -1325 | Yes | -1350 |
| 14 | -130 | -1480 | Yes | -1505 |
| 15 | -120 | -1625 | Yes | -1650 |
| 16 | -240 | -1890 | Yes | -1915 |
| 17 | +230 | -1685 | Yes | -1710 |
| 18 | -115 | -1825 | Yes | -1850 |
| 19 | +225 | -1625 | Yes | -1650 |
| 20 | -55 | -1705 | Yes | -1730 |
| 21 | -290 | -2020 | Yes | -2045 |
| 22 | +300 | -1745 | Yes | -1770 |
| 23 | -270 | -2040 | Yes | -2065 |
| 24 | +345 | -1720 | Yes | -1745 |
| 25 | -75 | -1820 | Yes | -1845 |
| 26 | -325 | -2170 | Yes | -2195 |

The balance never recovers to non-negative, so a fee is charged after every single operation.

<answer>
Answer: -2195
</answer>

---

### statetrack / robot-grid

**Prompt:** A robot stands on an 8x8 grid at cell (0, 0), facing north. Cells are
addressed (x, y): x increases to the east, y increases to the north. The grid wraps —
moving north from y=7 arrives at y=0, and mo...

**Response:**

Looking at this problem, I need to trace the robot's position and direction through each command on a wrapping 8×8 grid.

Starting: (0, 0), facing north

| Step | Command | Position | Facing |
|------|---------|----------|--------|
| 1 | L | (0,0) | west |
| 2 | L | (0,0) | south |
| 3 | F | (0,7) | south |
| 4 | F | (0,6) | south |
| 5 | R | (0,6) | west |
| 6 | L | (0,6) | south |
| 7 | F | (0,5) | south |
| 8 | F | (0,4) | south |
| 9 | F | (0,3) | south |
| 10 | R | (0,3) | west |
| 11 | F | (7,3) | west |
| 12 | F | (6,3) | west |
| 13 | F | (5,3) | west |
| 14 | L | (5,3) | south |
| 15 | R | (5,3) | west |
| 16 | R | (5,3) | south |
| 17 | L | (5,3) | east |
| 18 | R | (5,3) | south |
| 19 | F | (5,2) | south |
| 20 | R | (5,2) | west |
| 21 | F | (4,2) | west |
| 22 | R | (4,2) | south |
| 23 | F | (4,1) | south |
| 24 | L | (4,1) | east |
| 25 | R | (4,1) | south |
| 26 | R | (4,1) | west |
| 27 | L | (4,1) | south |
| 28 | L | (4,1) | east |
| 29 | R | (4,1) | south |
| 30 | R | (4,1) | west |
| 31 | L | (4,1) | south |
| 32 | F | (4,0) | south |
| 33 | F | (4,7) | south |
| 34 | F | (4,6) | south |
| 35 | L | (4,6) | east |
| 36 | F | (5,6) | east |
| 37 | L | (5,6) | north |
| 38 | F | (5,7) | north |
| 39 | F | (5,0) | north |
| 40 | R | (5,0) | east |

The robot never revisits (0,0) after the start.

<answer>
{"x": 5, "y": 0, "facing": "east", "returned_to_origin": 0}
</answer>

---

### statetrack / stack-machine

**Prompt:** A stack machine starts with an empty stack and supports five instructions:

- `PUSH n` — push the integer n on top of the stack
- `POP` — remove the top value and discard it
- `DUP` — push a copy of t...

**Response:**

<answer>
{"depth": 9, "top": 16, "sum": 77}
</answer>

---

### transformation / computed-table

**Prompt:** Below are 28 lines of `id,hours,rate`.

Produce a Markdown table with exactly these columns, in this order:

| ID | Hours | Rate | Pay | Band |

- `Pay` is hours multiplied by rate, rounded to two dec...

**Response:**

<answer>
| ID | Hours | Rate | Pay | Band |
|---|---|---|---|---|
| E2001 | 40 | 18.85 | 754.00 | mid |
| E2002 | 18 | 23.38 | 420.84 | low |
| E2003 | 33 | 47.64 | 1572.12 | high |
| E2004 | 29 | 18.25 | 529.25 | mid |
| E2005 | 23 | 41.83 | 962.09 | mid |
| E2006 | 45 | 37.04 | 1666.80 | high |
| E2007 | 31 | 47.49 | 1472.19 | mid |
| E2008 | 41 | 23.4 | 959.40 | mid |
| E2009 | 14 | 60.8 | 851.20 | mid |
| E2010 | 22 | 34.24 | 753.28 | mid |
| E2011 | 29 | 33.7 | 977.30 | mid |
| E2012 | 18 | 20.44 | 367.92 | low |
| E2013 | 32 | 31.26 | 1000.32 | mid |
| E2014 | 42 | 38.45 | 1614.90 | high |
| E2015 | 15 | 36.11 | 541.65 | mid |
| E2016 | 13 | 25.5 | 331.50 | low |
| E2017 | 36 | 22.03 | 793.08 | mid |
| E2018 | 39 | 31.29 | 1220.31 | mid |
| E2019 | 15 | 59.02 | 885.30 | mid |
| E2020 | 32 | 37.36 | 1195.52 | mid |
| E2021 | 20 | 24.76 | 495.20 | low |
| E2022 | 32 | 46.43 | 1485.76 | mid |
| E2023 | 44 | 59.68 | 2625.92 | high |
| E2024 | 32 | 33.06 | 1057.92 | mid |
| E2025 | 13 | 27.41 | 356.33 | low |
| E2026 | 33 | 20.69 | 682.77 | mid |
| E2027 | 26 | 59.72 | 1552.72 | high |
| E2028 | 26 | 36.01 | 936.26 | mid |
</answer>

---

### transformation / csv-to-json

**Prompt:** Convert the CSV below into a JSON array. Every row becomes one object with these keys:

- `id` — string, unchanged
- `name` — string, the first and last name joined with a single space
- `dept` — stri...

**Response:**

<answer>
[
  {"id":"E2001","name":"Marla Tanaka","dept":"Design","pay":759.60},
  {"id":"E2002","name":"Yuki Lindgren","dept":"Quality","pay":724.10},
  {"id":"E2003","name":"Ines Mbeki","dept":"Logistics","pay":792.00},
  {"id":"E2004","name":"Ada Vasquez","dept":"Design","pay":2291.08},
  {"id":"E2005","name":"Ada Baptiste","dept":"Quality","pay":1752.66},
  {"id":"E2006","name":"Ines Lindgren","dept":"Design","pay":413.28},
  {"id":"E2007","name":"Ruben Vasquez","dept":"Logistics","pay":1042.44},
  {"id":"E2008","name":"Devi Okonjo","dept":"Logistics","pay":839.54},
  {"id":"E2009","name":"Pilar Ferrers","dept":"Quality","pay":838.80},
  {"id":"E2010","name":"Yuki Ferrers","dept":"Field Service","pay":266.11},
  {"id":"E2011","name":"Marla Ferrers","dept":"Quality","pay":613.65},
  {"id":"E2012","name":"Ines Okonjo","dept":"Fabrication","pay":313.43},
  {"id":"E2013","name":"Ines Baptiste","dept":"Design","pay":367.32},
  {"id":"E2014","name":"Cyril Tanaka","dept":"Design","pay":1415.32},
  {"id":"E2015","name":"Ruben Vasquez","dept":"Field Service","pay":1565.22},
  {"id":"E2016","name":"Ines Vasquez","dept":"Quality","pay":1582.20},
  {"id":"E2017","name":"Ines Tanaka","dept":"Fabrication","pay":339.12},
  {"id":"E2018","name":"Ada Vasquez","dept":"Field Service","pay":2647.51},
  {"id":"E2019","name":"Yuki Baptiste","dept":"Field Service","pay":725.60},
  {"id":"E2020","name":"Ruben Okonjo","dept":"Fabrication","pay":744.66},
  {"id":"E2021","name":"Toma Tanaka","dept":"Field Service","pay":2191.98},
  {"id":"E2022","name":"Ruben Baptiste","dept":"Field Service","pay":1367.55},
  {"id":"E2023","name":"Osk Baptiste","dept":"Logistics","pay":1554.08},
  {"id":"E2024","name":"Cyril Ferrers","dept":"Fabrication","pay":791.70},
  {"id":"E2025","name":"Cyril Novak","dept":"Design","pay":1311.10},
  {"id":"E2026","name":"Toma Mbeki","dept":"Field Service","pay":845.07},
  {"id":"E2027","name":"Marla Ferrers","dept":"Fabrication","pay":399.56},
  {"id":"E2028","name":"Devi Lindgren","dept":"Design","pay":1610.10},
  {"id":"E2029","name":"Ada Lindgren","dept":"Field Service","pay":528.48},
  {"id":"E2030","name":"Cyril Ferrers","dept":"Field Service","pay":764.64}
]
</answer>

---

### transformation / schema-migration

**Prompt:** Below are 24 staff records, one per line:

    ID | NAME | DEPARTMENT | SITE | HOURS

Reorganise them into a JSON object keyed by department. Each department maps to an object
with:

- `headcount` — n...

**Response:**

<answer>
{"Logistics":{"headcount":6,"total_hours":201,"ids":["E2001","E2003","E2011","E2012","E2019","E2021"]},"Design":{"headcount":2,"total_hours":58,"ids":["E2002","E2022"]},"Field Service":{"headcount":7,"total_hours":150,"ids":["E2004","E2006","E2007","E2008","E2009","E2017","E2020"]},"Fabrication":{"headcount":5,"total_hours":152,"ids":["E2005","E2010","E2014","E2015","E2024"]},"Quality":{"headcount":4,"total_hours":109,"ids":["E2013","E2016","E2018","E2023"]},"busiest":"Logistics"}
</answer>

---

### writing / dialogue-only

**Prompt:** Write a scene of 250-350 words consisting entirely of dialogue between two characters.

Constraints:
- No narration, no action lines, no dialogue tags at all — not even "she said". Only the spoken lin...

**Response:**

<answer>
The kettle's on. You want tea or do you want to sit down.

I want to sit down. Tell me what he said.

He said a lot of things. Forty years of them. You don't need me to catalogue.

No. I mean this morning. St. Vincent's. At the very end.

The room. Right. You should have phoned. I was at the office, in the middle of—

You were at the office. I know. I'm not—

I'm not saying you should have. The water's boiling. I'm not standing here all morning. Blue cups or white ones.

The blue ones. The ones he liked. The chipped handle.

One set of blue cups. I'm pouring the tea. You're drinking it. Then we ring the solicitor.

The solicitor. The funeral. The eleven minutes I was alone in that room. I don't know if I should have let go when he grabbed my wrist or—

You held on. You held on and that's the whole of it.

Is it? He said something and I can't—

He said thank you. He meant it. That's the whole of it. Drink your tea.

He didn't say thank you.

He said, "You're the one I'm sorry I didn't call more." I don't know what that means. You were his daughter thirty-one years before I was born. You should know.

It means the morphine. He was tired. It means nothing, Ellie. Drink your tea.

It means he knew I was in the plastic chair. Six hours. He was sorry.

The tea's going bitter.

Pour it.

I poured it four minutes ago. You've been holding that cup. Haven't taken a sip.

Because you said that's the whole of it. And it isn't. You were at the office. He was in the bed. I was in the chair. He grabbed my wrist. He said I'm sorry I didn't call more. You weren't there to hold on. And I don't know if I held on long enough.
</answer>

---

### writing / explain-to-child

**Prompt:** Explain to a curious seven-year-old why the sky is blue.

Requirements:
- 120-200 words
- Use one concrete analogy a child could picture from everyday life
- No jargon: do not use the words "scatterin...

**Response:**

<answer>
You know how sunlight looks white? It's actually a mix of ALL the colors, like a rainbow hidden inside. Now here's the fun part: the air around us is full of super-tiny, invisible bits, way smaller than a grain of sand.

Think of it like this: imagine you're in a room full of tiny, tiny bouncy balls, and someone throws a handful of colored marbles through. The small blue marbles bounce off the bouncy balls and fly off in every direction—up, down, left, right. But the big red marbles are too heavy and just roll straight through.

That's what happens with sunlight and the air! The blue parts of the light bounce around in every direction, filling the whole sky with blue. The red and orange parts mostly keep going straight, which is why the sun itself looks yellow or orange.

So the sky isn't painted blue. It's full of tiny blue light bouncing around everywhere, just like those little marbles.

Next time you're outside, look at the sky right next to the sun and then look way up high. Do you notice the blue looks a little different in each spot?
</answer>

---

### writing / lighthouse

**Prompt:** Write a short story (around 300 words) about a lighthouse keeper who discovers that the light they tend every night is actually guiding something other than ships. The story should have a clear beginn...

**Response:**

<answer>
For thirty-one years, Maren climbed the one hundred and twelve steps each evening, lit the great Fresnel lens, and watched the beam sweep the black water. Ships. That was the purpose. The light kept them from the jagged teeth of the reef. She had inherited the duty from her mother, who had inherited it from her mother, and no one had ever asked what the light was truly for.

The truth came on a Tuesday in November, when the sea went still as poured glass.

No ships that night. No fog. Just the beam, cutting its slow arc across a water so calm it looked like dark oil. Maren stood at the gallery rail, her breath fogging in the cold, and watched the light sweep out, out, out—

And then the water *moved*.

Not a wave. A *heave*. The entire platform shuddered, and the lens swayed on its gimbal, throwing the beam in a lurching arc. Maren gripped the railing. Beneath the lighthouse, beneath the stone foundation she had walked over ten thousand times, something vast was *breathing*. A ridge of grey-green skin, wide as a cathedral nave, rose and fell in the black water. A blowhole the size of a cottage exhaled a plume of steam that caught the beam and scattered it into a thousand golden threads.

The creature turned its enormous eye toward the light. Blind. Clouded. Ancient. It had been following the beam for longer than the lighthouse had stood, longer than the island, longer than the name of the sea.

Maren understood then. The light was never a warning to ships.

It was a compass for the blind. A thread of gold in an endless dark, guiding the oldest living thing in the ocean so it would not drift into the crushing deep and die.

She climbed the one hundred and twelve steps the next evening, and the evening after that. And the one after.

The whale never once looked up. It didn't need to. It simply followed the light, the way a child follows a hand in the dark, and the lighthouse kept burning, and the sea kept turning, and no one ever asked what the light was truly for.
</answer>

---

### writing / technical-to-plain

**Prompt:** Rewrite the following notice so that a general audience can understand and act on it.

Original:
"Pursuant to the scheduled infrastructure maintenance window commencing 02:00 UTC on the 14th instant, ...

**Response:**

<answer>
Before 02:00 UTC on the 14th of this month — convert that to your local time zone — please save all in-progress work. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session at the time, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem that goes beyond the planned maintenance, report the issue through your standard support channels.
</answer>

---

### writing / unreliable-narrator

**Prompt:** Write a 300-400 word first-person account in which the narrator describes a day at their new job.

The narrator believes the day went well. The reader must finish the piece certain that it did not.

C...

**Response:**

<answer>
I've been waiting for a day like today for two years, and I'm glad it came.

The children came in at eight-fifteen and I had my little routine ready — the one where I ask them to find their spot, put their bags on the floor, and just breathe before we start. And they did it. Every single one. Not a whisper, not a fidget. I could feel the whole room settle into this quiet, shared stillness, and I thought, yes, this is what a classroom is supposed to feel like. They were so present with me. I could see it in their eyes.

Around ten, Lily started crying in the middle of the phonics group. She was really struggling to regulate, and I could tell the others were getting distracted. So I took her to the supply closet — it's small, enclosed, very quiet — and I sat with her for about twenty minutes. I told her she was safe, that no one was going to bother her. When I finally opened the door, she looked up and said, "I feel held." I teared up. I think she needed exactly that kind of contained, protected space, and I was able to give it to her.

By the afternoon, the energy in the room had shifted in the best way. The children were so focused, so careful with their movements. I could tell they trusted me to keep the space safe. I made sure to collect all their drawings and little personal things from their desks — I told them I'd keep everything in my office so nothing would get lost. They seemed so relieved. I want them to know their things are safe with me.

Mrs. Chen came in around three. She stood in the doorway for a long time, just looking at the room, then at me. She said, "You've certainly established yourself." I could hear the admiration in her voice. I think she'll be a wonderful colleague.

The headteacher pulled me in at four. She poured me a tea, sat me down, and said, "I want you to know we value your enthusiasm." Then she mentioned they'd be reviewing the day's events with the board tomorrow morning. I took that as a real compliment — they're clearly paying close attention to how I'm settling in. I'm already feeling like part of the team.

I'm going to sleep so well tonight.
</answer>

---

