# coding / lru-cache

**Prompt:** Implement a Python class called `LRUCache` with a fixed capacity that evicts the least recently used entry when full.

API:
- `LRUCache(capacity: int)` — create a cache holding at most `capacity` items
- `get(key)` — return the value, or `-1` if the key is absent. A successful get counts as a use.
- `put(key, value)` — insert or update. Updating an existing key counts as a use. If the cache is over capacity after inserting, evict the least recently used entry.

Requirements:
- Both `get` and `put` must run in O(1) average time
- Include type hints
- Do not use `functools.lru_cache`; implement the eviction yourself
- Explain in a comment which data structures give you the O(1) guarantee

## Repeat 1 (seed 0)

- Score: 9.75/10 — Correct O(1) doubly-linked-list + hash map implementation with type hints, proper LRU semantics, and clear O(1) explanation; minor clarity deduction for extra smoke-test but overall excellent.
- Tokens/s: 96.2
- Prefill tok/s: 1090.4
- TTFT: 12.78s
- Gen Time: 23.39s
- Output Tokens: 2222
- answer read from `answer_tag`

<details><summary>Reasoning (4398 chars, not scored)</summary>

The user wants me to implement an LRU Cache in Python with O(1) get and put operations.

The classic approach for O(1) LRU cache:
- A hash map (dict) for O(1) key lookup
- A doubly-linked list for O(1) insertion, deletion, and reordering to track access order

The doubly-linked list maintains the order of usage:
- Most recently used item is at the head
- Least recently used item is at the tail
- When we get or put an existing key, we move that node to the head
- When we put a new key and capacity is exceeded, we remove the tail node

Let me implement this:

1. A `Node` class for the doubly-linked list
2. The `LRUCache` class with:
   - `__init__(capacity)` - initialize dict, linked list with sentinels (head and tail)
   - `get(key)` - look up in dict, move node to front, return value or -1
   - `put(key, value)` - if key exists, update and move to front; if not, create node, add to front, if over capacity remove tail

Let me write this carefully:

```python
from __future__ import annotations
from typing import Optional


class _Node:
    """Doubly-linked list node."""
    __slots__ = ('key', 'value', 'prev', 'next')
    
    def __init__(self, key: int = 0, value: int = 0) -> None:
        self.key = key
        self.value = value
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class LRUCache:
    """
    O(1) LRU Cache.
    
    Data structures giving O(1) guarantee:
    - dict (hash map): O(1) average lookup by key to find the node.
    - Doubly-linked list with sentinel head/tail: O(1) insert at front,
      O(1) remove from tail (eviction), O(1) remove arbitrary node
      (we have a direct pointer via the dict).
    """
    
    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.cache: dict[int, _Node] = {}
        # Sentinel nodes to avoid edge-case checks
        self.head = _Node()  # dummy head
        self.tail = _Node()  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node: _Node) -> None:
        """Unlink node from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_front(self, node: _Node) -> None:
        """Insert node right after head (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def _move_to_front(self, node: _Node) -> None:
        """Move an existing node to the front (mark as most recently used)."""
        self._remove(node)
        self._add_to_front(node)
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_front(node)
        return node.value
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
        else:
            node = _Node(key, value)
            self.cache[key] = node
            self._add_to_front(node)
            if len(self.cache) > self.capacity:
                # Evict least recently used (tail.prev)
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
```

Wait, the type hints for key - the problem doesn't specify what type the key is. Let me use a generic or just use `int` as a common convention. Actually, looking at the problem statement, it just says `key` and `value` without specifying types. Let me use a generic type parameter or just use `int` for simplicity. Actually, let me use a type variable to be more general, or just use `int` since that's the most common convention for these problems.

Let me use a type variable for flexibility:

Actually, let me keep it simple and use `int` for keys and values since that's the standard LeetCode-style problem. But the problem doesn't specify, so let me use a generic approach or just `int`. I'll go with `int` for keys and values to keep it clean and standard.

Wait, let me re-read: "get(key) — return the value, or -1 if the key is absent." The -1 return suggests values are integers. Let me use `int` for both.

Let me also make sure the type hints are complete. I'll use `Optional` where needed.

Let me finalize my implementation.

</details>

**Answer (as scored):**

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

