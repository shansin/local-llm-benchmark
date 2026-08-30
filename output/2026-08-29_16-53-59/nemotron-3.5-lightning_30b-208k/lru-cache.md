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

- Score: 9.75/10 — Correct O(1) doubly-linked-list + hash map implementation with type hints, proper get/put semantics, eviction, and clear O(1) explanation; code passes all tests.
- Tokens/s: 299.9
- Prefill tok/s: 2419.5
- TTFT: 3.14s
- Gen Time: 4.96s
- Output Tokens: 1452
- answer read from `answer_tag`

<details><summary>Reasoning (3606 chars, not scored)</summary>

We need to implement LRUCache in Python with O(1) get and put. Use OrderedDict or doubly linked list + hash map. Since Python's OrderedDict has move_to_end and popitem, we can use that. But we need to implement ourselves? The requirement: "Do not use functools.lru_cache; implement the eviction yourself". It doesn't forbid using OrderedDict, but we can implement with a doubly linked list and a dict. However, using OrderedDict is simpler and still O(1) average. But to be safe, we can implement with a dict mapping key to node, and a doubly linked list with head/tail sentinel. We'll need Node class with key, value, prev, next. LRUCache has capacity, dict, and linked list. get: if key in dict, move node to front (most recently used), return value; else -1. put: if key exists, update value, move to front. If not, create node, add to front, add to dict. If size > capacity, remove tail node (least recently used) and delete from dict. We'll include type hints. Also comment about data structures: hash map for O(1) lookup, doubly linked list for O(1) insertion/removal at both ends.

We need to output final answer between <answer> and </answer> tags, with the code and comment. Ensure we follow instructions: "Everything inside those tags must be the answer itself, with no commentary about how you produced it." So just the code and comment.

Let's write the class.

```python
from __future__ import annotations  # for type hints if needed, but we can use standard.

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        # doubly linked list with sentinel head and tail
        self.head = Node(0, 0)  # dummy head
        self.tail = Node(0, 0)  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add(node)  # move to front (most recently used)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add(node)
        else:
            node = Node(key, value)
            self.cache[key] = node
            self._add(node)
            if len(self.cache) > self.capacity:
                # evict LRU
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]

    def _remove(self, node: Node) -> None:
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add(self, node: Node) -> None:
        # add right after head (most recently used)
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

class Node:
    def __init__(self, key: int, value: int):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
```

Add comment: "Hash map provides O(1) average-time lookup, while doubly linked list allows O(1) insertion and removal at both ends, enabling constant-time move-to-front and eviction of the least recently used entry."

Make sure to include type hints. The Node class could be inside or outside. We'll put it outside or inside? Usually we can define Node inside or outside. We'll define it outside for clarity. But we need to ensure the code is self-contained.

We'll output the answer.

</details>

**Answer (as scored):**

class Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: int, value: int):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """LRU cache with O(1) get and put.

    Data structures guaranteeing O(1):
    - Hash map (`dict`) provides average O(1) key lookup.
    - Doubly linked list (with sentinel head/tail) allows O(1) removal of any node and O(1) insertion at the front (most recently used) and at the back (least recently used for eviction).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: dict[int, Node] = {}
        # sentinel head and tail simplify edge cases
        self.head = Node(0, 0)  # dummy head, most recently used side
        self.tail = Node(0, 0)  # dummy tail, LRU side
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # move the accessed node to the front (most recently used)
        self._remove(node)
        self._add(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            # update counts as a use; move to front
            self._remove(node)
            self._add(node)
        else:
            node = Node(key, value)
            self.cache[key] = node
            self._add(node)
            # evict if over capacity
            if len(self.cache) > self.capacity:
                lru = self.tail.prev  # least recently used node
                self._remove(lru)
                del self.cache[lru.key]

    def _remove(self, node: Node) -> None:
        """Detach an existing node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add(self, node: Node) -> None:
        """Insert node right after head (most recently used position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

