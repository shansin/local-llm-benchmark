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

- Score: 10.0/10 — The response implements a correct O(1) LRU cache with dict + doubly linked list, proper get/put semantics, type hints, and an O(1) explanation; automated check about Python validity is overridden by the visible correct code.
- Tokens/s: 207.8
- Prefill tok/s: 2355.6
- TTFT: 0.13s
- Gen Time: 14.00s
- Output Tokens: 2883
- answer read from `answer_tag`

<details><summary>Reasoning (7394 chars, not scored)</summary>

The user wants me to implement an LRU cache in Python. Let me think about the design.

The classic solution uses a combination of a doubly linked list and a hash map (dictionary). This gives O(1) for both get and put.

Let me think about the design:

1. **Hash map (dict)**: Provides O(1) lookup by key. In Python 3.7+, dicts maintain insertion order, but for a true LRU we need to track usage order, so we use a doubly linked list.

2. **Doubly linked list**: Maintains the order of usage. The most recently used item is at the head (or tail), and the least recently used is at the other end. When we access or insert an item, we move it to the most-recently-used position. When we need to evict, we remove from the least-recently-used position.

The combination gives O(1) because:
- The dict gives O(1) key lookup
- The doubly linked list gives O(1) insertion, deletion, and reordering (since we have pointers to nodes)

Let me implement this.

I'll create a Node class with key, value, prev, next pointers.

I'll use sentinel head and tail nodes to simplify edge cases (avoiding None checks).

Let me write the code:

```python
from typing import Optional


class _Node:
    """A node in the doubly linked list."""
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: object, value: object) -> None:
        self.key = key
        self.value = value
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class LRUCache:
    """
    An LRU (Least Recently Used) cache with a fixed capacity.

    Data structures used for O(1) operations:
    - A Python `dict` maps keys to nodes, giving O(1) average lookup/insertion.
    - A doubly linked list maintains usage order. Because each node holds
      pointers to its neighbours, moving a node to the front or removing it
      from the back is O(1) given a reference to the node (which the dict
      already provides).

    Together these give O(1) average time for both `get` and `put`.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.size = 0
        self.cache: dict[object, _Node] = {}

        # Sentinel nodes: head marks the MRU end, tail marks the LRU end.
        self.head = _Node(None, None)
        self.tail = _Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Detach a node from the linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node: _Node) -> None:
        """Insert a node right after head (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: object) -> int:
        node = self.cache.get(key)
        if node is None:
            return -1
        # Mark as most recently used.
        self._remove(node)
        self._add_front(node)
        return node.value

    def put(self, key: object, value: object) -> None:
        if key in self.cache:
            # Update existing key and mark as most recently used.
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_front(node)
            return

        # Insert new node at the front.
        node = _Node(key, value)
        self.cache[key] = node
        self._add_front(node)
        self.size += 1

        # Evict the least recently used node (just before tail) if over capacity.
        if self.size > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
            self.size -= 1
```

Wait, the API says `get(key)` returns the value or -1. The value could be any type. Let me use `Any` for the value type but the return of get is `int` per the spec (returns -1 if absent). Actually, let me think. The spec says "return the value, or -1 if the key is absent." So the return type is the value type. But since we return -1 for absent, it's a bit ambiguous. Let me use `Any` for values to be flexible, but the get return type... Let me use `Any`.

Actually, let me reconsider. The typical LeetCode problem uses int keys and int values. But to be general, I'll use `Any`. Let me make the type hints reasonable.

Let me use `Any` for keys and values.

Actually, let me reconsider the return type of get. Since it returns the value or -1, and values could be ints, I'll annotate as `Any`.

Let me refine:

```python
from typing import Any, Optional


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Any, value: Any) -> None:
        self.key = key
        self.value = value
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class LRUCache:
    def __init__(self, capacity: int) -> None:
        ...
```

This looks good. Let me also add a small test/demo in the answer? The instructions say everything inside the tags must be the answer itself with no commentary about how I produced it. I think including a brief usage example is fine as it's part of the answer, but let me keep it minimal. Actually, let me include a small demonstration to show it works, but keep it clean.

Hmm, "no commentary about how you produced it" - I think a usage example is fine. But to be safe, I'll include the class and maybe a small test. Let me include a test at the bottom since it demonstrates correctness.

Actually, let me reconsider whether to include the test. The requirement is to implement the class. I'll include a small demonstration to verify. Let me keep it.

Let me double check the doubly linked list operations.

Sentinel setup:
```
head <-> tail
```

`_add_front` inserts after head:
```
node.next = head.next  # node.next = tail (initially)
node.prev = head
head.next.prev = node   # tail.prev = node
head.next = node
```
Result: head <-> node <-> tail. Correct.

`_remove`:
```
node.prev.next = node.next
node.next.prev = node.prev
```
Correct.

`get`: find node, remove, add front. Correct.

`put`: 
- If exists, update value, remove, add front.
- Else create node, add to cache, add front, increment size. If over capacity, evict tail.prev (LRU), remove from cache, decrement size.

This is correct.

Let me verify with an example:
- capacity 2
- put(1, 1): cache = {1: node1}, list: head <-> 1 <-> tail, size 1
- put(2, 2): cache = {1, 2}, list: head <-> 2 <-> 1 <-> tail, size 2
- get(1): returns 1, move 1 to front: head <-> 1 <-> 2 <-> tail
- put(3, 3): add 3, list: head <-> 3 <-> 1 <-> 2 <-> tail, size 3 > 2, evict tail.prev = 2. cache = {1, 3}, list: head <-> 3 <-> 1 <-> tail
- get(2): returns -1. Correct.

Great.

Let me finalize. I'll use `Any` for key and value types.

One more consideration: the `get` return type. Since it returns value or -1, I'll annotate return as `Any`.

Let me write the final answer.

I'll include a brief usage example to demonstrate. Actually, let me include it as it's helpful and part of a complete answer. But the instruction says "no commentary about how you produced it" - a usage example isn't commentary about production, it's part of the deliverable. I'll include a small test.

Let me write it cleanly.
</think>

</details>

**Answer (as scored):**

```python
from typing import Any, Optional


class _Node:
    """A node in the doubly linked list holding a (key, value) pair."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Any, value: Any) -> None:
        self.key = key
        self.value = value
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class LRUCache:
    """
    Least Recently Used (LRU) cache with a fixed capacity.

    How O(1) is achieved:
      * A Python `dict` maps each key to its linked-list node, giving O(1)
        average lookup, insertion and deletion by key.
      * A doubly linked list records usage order. Because every node stores
        pointers to its neighbours, moving a node to the most-recently-used
        end or unlinking the least-recently-used node is O(1) *given a node
        reference* — and the dict already hands us that reference.

    `get` and `put` therefore run in O(1) average time.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.size = 0
        self.cache: dict[Any, _Node] = {}

        # Sentinel head/tail nodes avoid None checks; head = MRU end,
        # tail = LRU end.
        self.head = _Node(None, None)
        self.tail = _Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node) -> None:
        """Detach a node from the list in O(1)."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node: _Node) -> None:
        """Insert `node` immediately after head (the MRU position) in O(1)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: Any) -> Any:
        node = self.cache.get(key)
        if node is None:
            return -1
        # A successful get counts as a use: promote to MRU end.
        self._remove(node)
        self._add_front(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        if key in self.cache:
            # Updating an existing key counts as a use.
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_front(node)
            return

        # Insert the new entry at the MRU end.
        node = _Node(key, value)
        self.cache[key] = node
        self._add_front(node)
        self.size += 1

        # Evict the least recently used node (just before the tail sentinel).
        if self.size > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
            self.size -= 1


if __name__ == "__main__":
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1          # cache = {1, 2}
    cache.put(3, 3)                   # evicts key 2 (LRU)
    assert cache.get(2) == -1         # returns -1 (key 2 was evicted)
    cache.put(4, 4)                   # evicts key 1 (LRU)
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4
    print("All tests passed.")

