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

- Score: 5.0/10 — NameError for _Node makes code non-functional; otherwise design is correct with O(1) structures and type hints.
- Tokens/s: 56.0
- Prefill tok/s: 1225.3
- TTFT: 0.30s
- Gen Time: 71.51s
- Output Tokens: 3987
- answer read from `answer_tag`

<details><summary>Reasoning (10978 chars, not scored)</summary>

We need answer user's coding request. Need produce final answer between  tags, complete. Need implement Python class LRUCache fixed capacity, evicts LRU. API get returns -1 absent, successful get counts as use. put insert/update, updating existing counts as use. If over capacity after inserting evict LRU. O(1) average. Type hints. No functools.lru_cache. Explain in comment which data structures give O(1). Need think first. We can implement with dict + doubly linked list. Need consider capacity maybe 0? Fixed capacity at most capacity items. If capacity <=0? Could handle by storing nothing, get -1, put no-op. But capacity int maybe positive. Need type hints. Could use typing? Python 3.9+ dict[str, Node]. Need maybe from __future__ import annotations? Not necessary. Need complete final answer only code? User asks implement class. We can include code block inside answer tags. Ensure no commentary outside? They said everything inside tags must be answer itself, no commentary about how produced. So final should be . Could include only code. Need maybe include comments. Let's design robust.

Class LRUCache:
- __init__(self, capacity: int) -> None:
  self._capacity = capacity
  self._cache: dict[int, _Node] = {}? Keys can be any hashable. Use dict[Any, _Node]. Values Any.
  self._head = _Node(None, None) sentinel? Use dummy head/tail.
  self._tail = _Node(None, None)
  self._head.next = self._tail; self._tail.prev = self._head
- _remove(node): O(1) unlink
- _add_front(node): insert after head (most recent)
- get(self, key: Any) -> Any:
  node = self._cache.get(key)
  if node is None: return -1
  self._move_to_front(node)
  return node.value
- put(self, key: Any, value: Any) -> None:
  if self._capacity <= 0: return? If capacity 0, no items. But if capacity negative? at most capacity items impossible. Could set capacity = max(0, capacity). But fixed capacity. If capacity <=0, do nothing.
  if key in self._cache:
     node = self._cache[key]
     node.value = value
     self._move_to_front(node)
  else:
     if len(self._cache) >= self._capacity:
        lru = self._tail.prev
        self._remove(lru)
        del self._cache[lru.key]
     node = _Node(key, value)
     self._cache[key] = node
     self._add_front(node)

Need type hints for internal Node. Could define nested class _Node with key: Any, value: Any, prev: Optional[_Node], next: Optional[_Node]. But self-referential type in nested class? Use from __future__ import annotations? Or use typing.Optional. In Python, nested class type annotation can be string? Simpler:
class _Node:
    def __init__(self, key: Any, value: Any) -> None:
       self.key = key
       self.value = value
       self.prev: Optional[_Node] = None
       self.next: Optional[_Node] = None
But _Node not defined yet at annotation evaluation? In Python, annotations in class body are evaluated at runtime unless from __future__ import annotations. For nested class, _Node is being defined, annotation Optional[_Node] would fail? Actually inside class body, name _Node is not yet bound until class creation completes. But annotations are evaluated in enclosing scope? Let's recall: In class body, annotations are evaluated in a separate namespace? Example:
class A:
    x: 'A' = None
works if string. Optional[A] inside class body? Name A not defined yet, so NameError. But if using typing.Optional[_Node] inside _Node class body, _Node not defined. Could use string annotations: self.prev: "_Node" = None. Or use from __future__ import annotations at top, then all annotations strings. But final code can include from __future__ import annotations. Or avoid annotations on prev/next? Requirement include type hints, but not necessarily every attribute. Could annotate methods. But better include. Use from __future__ import annotations.

Need maybe use collections? No.

Potential issue: get returns -1 if absent. If stored value is -1, get returns -1 ambiguous but API says return value or -1 if absent. Fine.

Need O(1) average comment: dict for O(1) lookup, doubly linked list for O(1) move/evict.

Need ensure updating existing key counts as use: move to front.

Need if capacity is 0: If put called, no insert. If get, -1. If capacity negative, treat as 0? Could do self._capacity = max(0, capacity). But if user expects fixed capacity negative? Not meaningful. Use if capacity <=0: self._capacity = 0. But type int.

Need maybe __len__? Not required. Could include.

Need final answer only code between tags. Could include docstring. Need no extra explanation outside tags. The code comment explains data structures.

Let's consider edge cases:
- Duplicate put: update value and move to front.
- Eviction: if len >= capacity before insert. If capacity 0, len 0 >=0, would try tail.prev which is head? We guard capacity <=0 return.
- If capacity positive, sentinel tail.prev is most? We add front after head. LRU is tail.prev. Good.
- _move_to_front: remove then add_front. If node already front, remove/add okay.
- _remove: update prev/next. Need handle sentinels? We only remove real nodes. But can implement generic.

Potential type hints with Any from typing. Use:
from __future__ import annotations
from typing import Any, Optional

class LRUCache:
    """..."""
    class _Node:
       def __init__(self, key: Any, value: Any) -> None:
          self.key = key
          self.value = value
          self.prev: Optional[_Node] = None
          self.next: Optional[_Node] = None

But nested class with from __future__ annotations: annotations are strings, okay. However Optional[_Node] in class body with future? It becomes string, no evaluation. Good.

But nested class name _Node inside LRUCache. In method type hints, use LRUCache._Node? Could just use _Node in local? In class body, _Node defined. For method return types maybe None. For private methods:
    def _remove(self, node: _Node) -> None:
But _Node is in class namespace, okay after nested class defined. With future annotations, okay.

Need maybe use list? No.

Could implement with OrderedDict? Requirement implement eviction yourself, do not use functools.lru_cache. OrderedDict is standard library but maybe they want own data structures. Better not use OrderedDict. Use dict + linked list.

Need average O(1). Dict average O(1). Linked list operations O(1).

Let's craft final code:


Need consider if key can be None? dict key None okay. Node key None for sentinels. If user stores key None, sentinel key None but not in dict. Fine.

Potential issue: _remove sets node.prev/next None. If node is sentinel? We don't remove sentinels. Good.

Potential issue: If capacity is negative, max(0). Good.

Potential issue: If value is -1, get returns -1 for present and absent. API says return value or -1 if absent. That's inherent. Could use sentinel? No.

Need maybe type hint for get return Any. Good.

Could there be issue with nested class annotation Optional[_Node] under from __future__? It's fine. But if no future, would fail. We include.

Need final only answer tags. Ensure no markdown? They said put complete final answer between  tags. Could include code block inside. It's okay. But maybe they want raw. We'll output:


Need no commentary. Good.

Double-check O(1) average: dict operations average O(1). Linked list pointer updates constant.

Could include comment specifically "which data structures give you O(1) guarantee". Docstring comment does. Maybe also inline comment. Good.

Let's think if there is any hidden requirement: "Implement a Python class called LRUCache with a fixed capacity that evicts the least recently used entry when full." API. "Both get and put must run in O(1) average time" "Include type hints" "Do not use functools.lru_cache; implement eviction yourself" "Explain in a comment which data structures give you the O(1) guarantee". We satisfy.

Could they test with capacity 1, put(1,1), put(2,2) evicts 1. get(1) -1, get(2) 2. Good.

Let's simulate:
init head<->tail.
put 1: len 0 <1, node1 add front: head.next=node1, node1.prev=head, node1.next=tail, tail.prev=node1. dict {1:node1}
put 2: len 1 >=1, lru=tail.prev=node1, remove: node1.prev=head, node1.next=tail => head.next=tail, tail.prev=head. del. add node2. Good.
get 2: move_to_front: remove node2: head.next=tail, tail.prev=head; add front. Good.

If update existing: put(2,3): existing, value=3, move. Good.

If capacity 0: put returns, dict empty. get returns -1. Good.

Could there be issue with len(self._cache) >= self._capacity when capacity positive but dict length maybe less due to bug? No.

Need maybe use `self._cache: dict[Any, _Node]` requires Python 3.9. With future annotations, okay. But runtime type of dict? It's just annotation. If Python <3.9, dict generic not supported at runtime? With future annotations, annotations are strings, so okay even 3.7? But `dict[Any, _Node]` as string not evaluated. However if no future, Python 3.8 would fail. We include future. But `from __future__ import annotations` available 3.7. Good.

But nested class `_Node` annotation `Optional[_Node]` with future okay. But `self._cache: dict[Any, _Node] = {}` in __init__ with future okay. Good.

Could use `typing.Dict` for older? Not necessary. But if environment Python 3.8, future makes okay. If Python 3.7, okay. If Python 3.11, okay.

Need maybe `Optional` from typing. Good.

One subtlety: In nested class `_Node`, with `from __future__ import annotations`, the annotation `Optional[_Node]` is stored as string, but the class body still executes `self.prev = None`. No issue. But if some tool introspects annotations, it may resolve. Fine.

Another subtlety: `self._head = _Node(None, None)` type: _Node expects Any, okay.

Could include `__slots__`? Not necessary.

Need final answer no analysis. Good.

Let's consider if they expect no imports? Type hints can use built-in. But imports okay.

Could there be a problem with returning -1 as int but value Any. If key absent, return -1. If value type not int, okay.

Could they test `isinstance`? No.

Could they test thread safety? No.

Could they test capacity as float? Type int. We max(0, capacity) if float? max works, but len compare with float? If capacity 2.5, len >=2.5 weird. But type int. Could cast? Not needed.

Could they test negative capacity? We handle.

Could they test updating after eviction? Good.

Need maybe ensure "fixed capacity" not resizable. We don't expose capacity change.

Let's think about alternative: Use `collections.OrderedDict` would be simpler but requirement implement eviction yourself. Our linked list is own.

Need maybe comment "A dict maps keys to nodes for O(1) lookup; a doubly linked list maintains recency order for O(1) move/evict." Good.

Final should be exactly answer tags. Could include newline. Ensure closing tag.

One more thought: The user says "Think first if you need to, then put your complete final answer between  tags." We as Qwen final can just answer. No need mention thinking. Good.

Let's produce final.
</think>

</details>

**Answer (as scored):**

from __future__ import annotations

from typing import Any, Optional


class LRUCache:
    """
    Fixed-capacity LRU cache.

    O(1) average time guarantee:
    - dict: O(1) average lookup, update, and deletion by key
    - doubly linked list: O(1) move-to-most-recent and remove-least-recent
    """

    class _Node:
        def __init__(self, key: Any, value: Any) -> None:
            self.key = key
            self.value = value
            self.prev: Optional[_Node] = None
            self.next: Optional[_Node] = None

    def __init__(self, capacity: int) -> None:
        self._capacity: int = max(0, capacity)
        self._cache: dict[Any, _Node] = {}

        # Sentinel nodes avoid special-casing head/tail during removal.
        self._head = _Node(None, None)
        self._tail = _Node(None, None)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _add_front(self, node: _Node) -> None:
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: _Node) -> None:
        self._remove(node)
        self._add_front(node)

    def get(self, key: Any) -> Any:
        node = self._cache.get(key)
        if node is None:
            return -1

        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        if self._capacity == 0:
            return

        existing = self._cache.get(key)
        if existing is not None:
            existing.value = value
            self._move_to_front(existing)
            return

        if len(self._cache) >= self._capacity:
            lru = self._tail.prev
            self._remove(lru)
            del self._cache[lru.key]

        node = _Node(key, value)
        self._cache[key] = node
        self._add_front(node)

    def __len__(self) -> int:
        return len(self._cache)

