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

- Score: 1.0/10 — The response contains no valid Python code per automated check, so it fails all functional requirements.
- Tokens/s: 206.4
- Prefill tok/s: 2443.6
- TTFT: 0.14s
- Gen Time: 21.28s
- Output Tokens: 4363
- answer read from `answer_tag`

<details><summary>Reasoning (14422 chars, not scored)</summary>

Let me think about this topological sort problem.

We have a directed graph as a dict mapping each node to a list of nodes it points to. We need to return a list where every node appears before the nodes it points to.

Requirements:
- Nodes that appear only as targets must be included
- Handle empty graph
- Type hints
- O(V + E)
- If cycle, raise ValueError with message naming at least one node in the cycle

Let me use DFS-based topological sort (Kahn's algorithm or DFS).

Let me use DFS approach with three states: unvisited, in-progress (visiting), done. If we encounter a node that's in-progress, we have a cycle.

Let me write the code:

```python
from typing import Dict, List, Any

def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    # Collect all nodes (keys and targets)
    all_nodes = set(graph.keys())
    for targets in graph.values():
        all_nodes.update(targets)
    
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in all_nodes}
    result: List[Any] = []
    
    def visit(node: Any) -> None:
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if color[neighbor] == GRAY:
                raise ValueError(f"Cycle detected involving node '{neighbor}'")
            if color[neighbor] == WHITE:
                visit(neighbor)
        color[node] = BLACK
        result.append(node)
    
    for node in all_nodes:
        if color[node] == WHITE:
            visit(node)
    
    return result
```

Wait, there's a subtlety with recursion depth for large graphs, but for O(V+E) it's fine. Let me consider iterative to avoid recursion limits, but recursion is simpler. Let me think about whether recursion is acceptable. The requirement is O(V+E) which recursion satisfies.

Actually, let me reconsider the cycle detection message. The message should name at least one node involved in the cycle. When we find a GRAY node, that node is part of the cycle. Good.

But wait, the recursion approach with `all_nodes` being a set means iteration order is non-deterministic. That's fine for correctness but the example expects a specific order. Let me check the example:

`topo_sort({"a": ["b"], "b": ["c"], "c": []})` => `["a", "b", "c"]`

With my DFS approach, starting from 'a' (if it's first in iteration), we visit a, then b, then c. Result would be [c, b, a] reversed... wait no.

Let me trace: visit(a): color[a]=GRAY, neighbor b. visit(b): color[b]=GRAY, neighbor c. visit(c): color[c]=GRAY, no neighbors, color[c]=BLACK, result=[c]. Back to b: color[b]=BLACK, result=[c,b]. Back to a: color[a]=BLACK, result=[c,b,a].

So result = [c, b, a]. But expected is [a, b, c].

Hmm, so the DFS post-order gives reverse topological order. I need to reverse it, OR use pre-order.

Actually, the standard DFS topological sort appends in post-order and then reverses. Let me reverse the result.

Wait, but if I reverse [c, b, a] I get [a, b, c]. That matches!

But the iteration order of the set matters. If I iterate over all_nodes in a different order, I might get a different result. To match the example deterministically, I should iterate in a consistent order.

Actually, the example just shows one valid output. But to be safe and match, let me think about what order to iterate.

Hmm, but sets don't preserve order. Let me use a different approach to preserve insertion order or sort.

Actually, let me reconsider. The problem says "Return a list of all nodes in an order such that every node appears before the nodes it points to." So any valid topological order is acceptable. But the example shows ["a", "b", "c"].

To make it deterministic and match the example, I could iterate over nodes in the order they first appear (keys first in insertion order, then targets). Let me build an ordered list of all nodes.

Let me use an ordered approach:

```python
def topo_sort(graph):
    # Build ordered list of all nodes
    seen = []
    seen_set = set()
    for k, v in graph.items():
        if k not in seen_set:
            seen.append(k)
            seen_set.add(k)
        for t in v:
            if t not in seen_set:
                seen.append(t)
                seen_set.add(t)
    ...
```

Then iterate over `seen` in order.

Let me trace the example: graph = {"a": ["b"], "b": ["c"], "c": []}
- k="a", v=["b"]: seen=["a"], then "b" not seen, seen=["a","b"]
- k="b", v=["c"]: "b" seen, "c" not seen, seen=["a","b","c"]
- k="c", v=[]: "c" seen

Iterate: visit("a"): GRAY, neighbor "b". visit("b"): GRAY, neighbor "c". visit("c"): GRAY, no neighbors, BLACK, result=["c"]. Back: "b" BLACK, result=["c","b"]. Back: "a" BLACK, result=["c","b","a"]. Reverse: ["a","b","c"]. 

Now let me think about whether to reverse or use Kahn's algorithm.

Actually, let me reconsider. There might be an issue with the recursion. Let me use Kahn's algorithm instead which is iterative and naturally gives the right order.

Kahn's algorithm:
1. Compute in-degree for each node.
2. Start with nodes that have in-degree 0.
3. Process queue, decrement in-degrees of neighbors, add to queue when in-degree becomes 0.
4. If processed count < total nodes, there's a cycle.

Let me trace the example:
- Edges: a->b, b->c, c->(none)
- In-degrees: a:0, b:1, c:1
- Queue: [a]
- Process a: result=[a], decrement b: b:0, queue=[b]
- Process b: result=[a,b], decrement c: c:0, queue=[c]
- Process c: result=[a,b,c], queue=[]
- Processed 3 = total 3. No cycle.
- Return [a,b,c]. 

This matches directly without reversing. And Kahn's is iterative, avoiding recursion limits.

Let me implement Kahn's:

```python
from typing import Dict, List, Any
from collections import deque, defaultdict

def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    # Collect all nodes in a deterministic order
    all_nodes: List[Any] = []
    node_set: set = set()
    for k, v in graph.items():
        if k not in node_set:
            all_nodes.append(k)
            node_set.add(k)
        for t in v:
            if t not in node_set:
                all_nodes.append(t)
                node_set.add(t)
    
    # Compute in-degrees
    in_degree: Dict[Any, int] = {node: 0 for node in all_nodes}
    for k, v in graph.items():
        for t in v:
            in_degree[t] += 1
    
    # Initialize queue with in-degree 0 nodes
    queue: deque = deque(node for node in all_nodes if in_degree[node] == 0)
    
    result: List[Any] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(result) != len(all_nodes):
        # Find a node in the cycle
        cycle_node = next(node for node in all_nodes if in_degree[node] > 0)
        raise ValueError(f"Cycle detected involving node '{cycle_node}'")
    
    return result
```

Wait, there's a subtle issue with in_degree computation. If a node appears multiple times as a target, in_degree would be incremented multiple times. But also, when processing, we decrement for each edge. So as long as edges are counted consistently, it works. But if there are duplicate edges (a node listed twice in a target list), the in-degree would be 2 and we'd decrement 2 times. That's consistent. But it could cause issues if... actually it's fine as long as consistent.

Hmm, but actually there's a subtle issue. What if a node points to itself? a->a. Then in_degree[a] = 1 (from the self-loop). It never reaches 0, so it stays in the cycle. Good, that's detected.

Let me also double check: the in_degree dict is initialized for all nodes. When we do `in_degree[t] += 1` for targets, that's correct.

Actually wait, there's a potential issue. What if a target `t` is not in `all_nodes`? No, we added all targets to all_nodes. Good.

Let me reconsider the graph.get(node, []) - since all keys are in graph, this is just graph[node]. But using .get is safe.

Let me verify the cycle detection. If there's a cycle, the nodes in the cycle will always have in_degree > 0 because at least one edge into them is never "removed" (since the source never gets processed). So `next(node for node in all_nodes if in_degree[node] > 0)` will find a node in the cycle. Good.

Let me trace a cycle: a->b, b->a.
- all_nodes = [a, b]
- in_degree: a:1 (from b->a), b:1 (from a->b)
- queue: [] (no in-degree 0 nodes)
- result: []
- len(result)=0 != 2, so find node with in_degree > 0: a. Raise ValueError "Cycle detected involving node 'a'". 

Empty graph: topo_sort({})
- all_nodes = []
- in_degree = {}
- queue = []
- result = []
- len(result)=0 == len(all_nodes)=0. Return []. 

Now let me think about the type hints. The graph is Dict[Any, List[Any]]. Return List[Any].

Let me also consider: should I use `defaultdict` for in_degree? No, I initialize all nodes to 0, so regular dict works.

Actually, let me reconsider the in_degree computation more carefully. I iterate over graph.items() and for each target increment. But what about nodes that are only targets? They're in all_nodes with in_degree 0 initially, then incremented. Good.

Let me also think about whether the order of iteration matters for matching the example. With Kahn's, I process nodes in the order they appear in all_nodes (which is insertion order). For the example, that gives [a, b, c]. 

But wait, the queue initialization: `deque(node for node in all_nodes if in_degree[node] == 0)`. For the example, only 'a' has in_degree 0, so queue = [a]. Good.

Let me finalize. I think Kahn's algorithm is clean and iterative.

Actually, let me reconsider one thing about the message. The requirement says "raise ValueError with a message naming at least one node involved in the cycle." My message includes the node name. Good.

Let me also think about whether I should handle the case where graph values might contain nodes not... no, all handled.

Let me write the final version. I'll use type hints properly.

```python
from typing import Any, Dict, List
from collections import deque


def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    """Perform a topological sort of a directed graph.

    Args:
        graph: A dict mapping each node to the list of nodes it points to.

    Returns:
        A list of all nodes such that every node appears before the nodes
        it points to.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    # Collect every node (keys and targets) preserving first-seen order.
    ordered: List[Any] = []
    seen: set = set()
    for node, targets in graph.items():
        if node not in seen:
            ordered.append(node)
            seen.add(node)
        for target in targets:
            if target not in seen:
                ordered.append(target)
                seen.add(target)

    # Compute in-degrees.
    in_degree: Dict[Any, int] = {node: 0 for node in ordered}
    for targets in graph.values():
        for target in targets:
            in_degree[target] += 1

    # Start from nodes with no incoming edges.
    queue: deque = deque(n for n in ordered if in_degree[n] == 0)
    result: List[Any] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, ()):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(ordered):
        cycle_node = next(n for n in ordered if in_degree[n] > 0)
        raise ValueError(
            f"Graph contains a cycle; node '{cycle_node}' is involved."
        )

    return result
```

This looks good. Let me double-check the O(V+E) requirement. 
- Collecting nodes: O(V + E)
- In-degree: O(V + E)
- Queue processing: each node processed once, each edge traversed once: O(V + E)
- Cycle check: O(V)
Total: O(V + E). 

One concern: `graph.get(node, ())` - since all keys are in graph, this returns graph[node]. Fine.

Let me reconsider whether there could be an issue with duplicate targets causing incorrect in-degree. Suppose a->b, a->b (b listed twice). Then in_degree[b] = 2. When processing a, we decrement b twice (once for each occurrence in a's list), so b reaches 0. Consistent. But this is a multigraph scenario. The problem probably assumes simple graphs, but it works either way.

Actually wait, there's a subtle bug. If b is listed twice in a's targets, in_degree[b] = 2. When we process a, we iterate over graph.get(a) = [b, b], decrementing in_degree[b] to 1 then 0. So b gets added to queue when it reaches 0. Correct. Good.

I'm confident in this solution. Let me also make sure the example works exactly.

topo_sort({"a": ["b"], "b": ["c"], "c": []}):
- ordered: 
  - node="a", targets=["b"]: "a" not seen -> ordered=["a"], seen={"a"}. "b" not seen -> ordered=["a","b"], seen={"a","b"}.
  - node="b", targets=["c"]: "b" seen. "c" not seen -> ordered=["a","b","c"], seen={"a","b","c"}.
  - node="c", targets=[]: "c" seen.
- in_degree: {a:0, b:0, c:0}
  - targets ["b"]: in_degree[b] += 1 -> b:1
  - targets ["c"]: in_degree[c] += 1 -> c:1
  - targets []: nothing
  - in_degree = {a:0, b:1, c:1}
- queue: [a] (only a has in_degree 0)
- Process a: result=[a]. neighbors ["b"]: in_degree[b]=0 -> queue=[b]
- Process b: result=[a,b]. neighbors ["c"]: in_degree[c]=0 -> queue=[c]
- Process c: result=[a,b,c]. neighbors []: nothing.
- len(result)=3 == len(ordered)=3.
- Return [a,b,c]. 

Matches expected output.

I'm confident. Let me write the final answer.

Actually, let me reconsider the type hint for the return and the deque. The deque is internal, so it's fine. Let me make sure imports are correct.

Final answer below.
</think>

## Reasoning

I'll use **Kahn's algorithm** (BFS-based), which is naturally iterative (avoiding recursion limits) and runs in O(V + E):

1. **Collect all nodes** — both keys and targets — preserving first-seen order for deterministic output.
2. **Compute in-degrees** for every node.
3. **Process** nodes with in-degree 0, decrementing neighbors' in-degrees and enqueueing any that reach 0.
4. **Cycle detection** — if not all nodes are emitted, some remain with in-degree > 0 (part of a cycle); report one such node.

This handles the empty graph (returns `[]`), includes target-only nodes, and is O(V + E).

**Trace of example:** `{"a":["b"], "b":["c"], "c":[]}` → in-degrees `{a:0, b:1, c:1}` → queue `[a]` → emit `a`, `b`, `c` → `[a, b, c]` ✓

</details>

**Answer (as scored):**

```python
from typing import Any, Dict, List
from collections import deque


def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    """Perform a topological sort of a directed graph.

    Args:
        graph: A dict mapping each node to the list of nodes it points to.

    Returns:
        A list of all nodes such that every node appears before the nodes
        it points to.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    # Collect every node (keys and targets), preserving first-seen order.
    ordered: List[Any] = []
    seen: set = set()
    for node, targets in graph.items():
        if node not in seen:
            ordered.append(node)
            seen.add(node)
        for target in targets:
            if target not in seen:
                ordered.append(target)
                seen.add(target)

    # Compute in-degrees.
    in_degree: Dict[Any, int] = {node: 0 for node in ordered}
    for targets in graph.values():
        for target in targets:
            in_degree[target] += 1

    # Start from nodes with no incoming edges.
    queue: deque = deque(n for n in ordered if in_degree[n] == 0)
    result: List[Any] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, ()):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(ordered):
        cycle_node = next(n for n in ordered if in_degree[n] > 0)
        raise ValueError(
            f"Graph contains a cycle; node '{cycle_node}' is involved."
        )

    return result

