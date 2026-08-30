# coding / topological-sort

**Prompt:** Write a Python function called `topo_sort` that performs a topological sort of a directed graph.

Signature: `topo_sort(graph)` where `graph` is a dict mapping each node to a list of the nodes it points to (its dependencies-after, i.e. edges go from key to each element).

Return a list of all nodes in an order such that every node appears before the nodes it points to.

If the graph contains a cycle, raise `ValueError` with a message naming at least one node involved in the cycle.

Requirements:
- Nodes that appear only as targets (never as keys) must still be included in the output
- Handle the empty graph
- Include type hints
- The algorithm must be O(V + E)

Example:
  topo_sort({"a": ["b"], "b": ["c"], "c": []}) => ["a", "b", "c"]

## Repeat 1 (seed 0)

- Score: 10.0/10 — Implementation uses Kahn's algorithm O(V+E), includes all nodes, type hints, raises ValueError naming a node in cycle, handles empty graph, and passes all automated tests.
- Tokens/s: 95.1
- Prefill tok/s: 1150.5
- TTFT: 16.09s
- Gen Time: 20.19s
- Output Tokens: 1891
- answer read from `answer_tag`

<details><summary>Reasoning (4993 chars, not scored)</summary>

Let me think through this carefully.

We need a topological sort function where:
- `graph` is a dict mapping each node to a list of nodes it points to (edges go from key to each element)
- Return a list where every node appears BEFORE the nodes it points to
- If there's a cycle, raise ValueError naming at least one node in the cycle
- Nodes that appear only as targets (never as keys) must be included
- Handle empty graph
- Type hints
- O(V + E)

Wait, let me re-read: "Return a list of all nodes in an order such that every node appears before the nodes it points to."

So if edge is a -> b, then a must appear before b in the output.

Example: `topo_sort({"a": ["b"], "b": ["c"], "c": []})` => `["a", "b", "c"]`

So a -> b -> c, and the output is a, b, c. That confirms: source before target.

This is a standard topological sort. Let me think about the algorithm.

**Kahn's algorithm (BFS-based):**
1. Compute in-degree of each node
2. Start with nodes that have in-degree 0
3. Process them, reducing in-degrees of their neighbors
4. If we can't process all nodes, there's a cycle

Wait, but the edges go FROM key TO each element. So if "a": ["b"], that means a -> b. In a topological sort where a must come before b, a has no incoming edges (in-degree 0) and b has in-degree 1.

So in-degree: for each edge u -> v, v's in-degree increases by 1.

Kahn's algorithm:
1. Collect all nodes (keys + all values in lists)
2. Compute in-degree for each node
3. Queue all nodes with in-degree 0
4. Process: pop node, add to result, for each neighbor, decrement in-degree, if 0 add to queue
5. If result length != total nodes, there's a cycle

For the cycle detection with a named node: if there are remaining nodes (in-degree > 0), pick one of them.

Let me think about the in-degree computation:
- For each key `u` in graph, for each `v` in graph[u], v's in-degree += 1
- Nodes that are only targets (never keys) will have in-degree > 0 but won't be in the graph dict as keys. We need to include them.

Let me code this up:

```python
from typing import Any

def topo_sort(graph: dict[Any, list[Any]]) -> list[Any]:
    # Collect all nodes
    nodes: set[Any] = set(graph.keys())
    for neighbors in graph.values():
        nodes.update(neighbors)
    
    # Compute in-degrees
    in_degree: dict[Any, int] = {node: 0 for node in nodes}
    for u, neighbors in graph.items():
        for v in neighbors:
            in_degree[v] += 1
    
    # Initialize queue with in-degree 0 nodes
    from collections import deque
    queue = deque(node for node in nodes if in_degree[node] == 0)
    
    result: list[Any] = []
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    if len(result) != len(nodes):
        # Find a node in the cycle
        for node in nodes:
            if in_degree[node] > 0:
                raise ValueError(f"Cycle detected involving node: {node}")
        # This shouldn't happen but just in case
        raise ValueError("Cycle detected")
    
    return result
```

Wait, let me verify with the example:
- graph = {"a": ["b"], "b": ["c"], "c": []}
- nodes = {"a", "b", "c"}
- in_degree: a=0, b=1 (from a->b), c=1 (from b->c)
- queue starts with: ["a"]
- Process a: result=[a], b's in-degree becomes 0, queue=["b"]
- Process b: result=[a,b], c's in-degree becomes 0, queue=["c"]
- Process c: result=[a,b,c]
- len(result) == 3 == len(nodes) ✓

Cycle example: {"a": ["b"], "b": ["a"]}
- nodes = {"a", "b"}
- in_degree: a=1, b=1
- queue starts empty
- result = []
- len(result) = 0 != 2
- Find node with in_degree > 0: "a" (or "b")
- Raise ValueError ✓

Empty graph: {}
- nodes = set()
- in_degree = {}
- queue = empty
- result = []
- len(result) == 0 == len(nodes) ✓
- Return []

Node only as target: {"a": ["b"]}
- nodes = {"a", "b"}
- in_degree: a=0, b=1
- queue: ["a"]
- Process a: result=[a], b's in-degree becomes 0, queue=["b"]
- Process b: result=[a, b]
- ✓

This looks correct. Let me also think about whether the order is deterministic. With a set, the order of nodes with in-degree 0 might vary. But the problem doesn't require a specific order, just a valid topological ordering. So this is fine.

Actually, wait - I should double-check: the problem says "every node appears before the nodes it points to". So for edge a->b, a must be before b. My algorithm does exactly that. ✓

Let me also make sure the type hints are proper. I'll use `dict[Any, list[Any]]` or maybe more specific. Actually, let me use a type variable for better typing.

Actually, let me keep it simple with `Any` since the problem doesn't specify a particular type for nodes. Or I could use a TypeVar.

Let me use a TypeVar for cleaner type hints:

```python
from typing import TypeVar, Any
from collections import deque

T = TypeVar('T')

def topo_sort(graph: dict[T, list[T]]) -> list[T]:
```

This is cleaner. Let me finalize.

</details>

**Answer (as scored):**

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

