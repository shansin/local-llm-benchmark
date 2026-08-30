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

- Score: 8.0/10 — DFS recursion fails on large chains causing RecursionError, otherwise correct topological order, cycle detection, type hints, and node inclusion are satisfied.
- Tokens/s: 280.3
- Prefill tok/s: 2409.8
- TTFT: 49.23s
- Gen Time: 50.07s
- Output Tokens: 14001
- answer read from `answer_tag`

<details><summary>Reasoning (52811 chars, not scored)</summary>

We need to write a Python function `topo_sort(graph)` that performs topological sort of a directed graph. The graph is a dict mapping each node to a list of nodes it points to (dependencies-after). So edges go from key to each element. Return a list of all nodes in an order such that every node appears before the nodes it points to. If the graph contains a cycle, raise ValueError with a message naming at least one node involved in the cycle.

Requirements:
- Nodes that appear only as targets (never as keys) must still be included in the output.
- Handle the empty graph.
- Include type hints.
- The algorithm must be O(V + E).

Example: topo_sort({"a": ["b"], "b": ["c"], "c": []}) => ["a", "b", "c"].

We need to implement topological sort. Since nodes that appear only as targets must be included, we need to collect all nodes: keys and all values in adjacency lists. Then we can do Kahn's algorithm (BFS-based) or DFS-based. We also need to detect cycles and raise ValueError with a message naming at least one node involved in the cycle.

We need to output a list of all nodes in topological order. If there's a cycle, we need to identify at least one node in the cycle. The error message should name at least one node involved in the cycle.

We need to ensure O(V+E) time.

Let's think about algorithm: Kahn's algorithm (in-degree based) naturally detects cycles: if after processing all nodes with in-degree 0, there are still nodes left, there is a cycle. But we need to name at least one node involved in the cycle. We could keep track of nodes that remain, or we could use DFS with coloring (white, gray, black) to detect back edges and then extract a cycle. Since we need to name at least one node, we can just find any node that is part of a cycle. In Kahn's algorithm, we could also find a node that remains with positive in-degree, but that doesn't directly give a cycle node. However, we can modify Kahn's to also record the cycle. Alternatively, we can do DFS and when we encounter a gray node (currently in recursion stack), we have found a cycle. Then we can backtrack to extract a cycle path. But we need to output a message naming at least one node involved in the cycle. So we can just record the node where the back edge was found, or the node that caused the cycle.

Simpler: Use DFS-based topological sort with cycle detection. We'll maintain a set of visited nodes, and a set of nodes in the current recursion stack. When we encounter a node that is in the recursion stack, we have a cycle. We can then extract a cycle by following parent pointers or just record the node that triggered the cycle. Since we need to name at least one node, we can just record the node that we are currently visiting when we find the back edge. But we need to ensure the error message includes a node involved in the cycle. We can just say "Cycle detected involving node X". But we need to be sure X is indeed in a cycle. If we find a back edge from u to v, and v is in the recursion stack, then both u and v are in the cycle. So we can raise ValueError with message like f"Graph contains a cycle involving node {v}" or something.

But we must also include all nodes that appear only as targets. So we first collect all nodes: all keys and all values in adjacency lists. Then we build the graph. We'll need to handle nodes that have no outgoing edges (they might not be keys, but they are targets). In the dict, they might not be present as keys. So we need to ensure they are included in the output.

Algorithm steps:

1. Collect all nodes: nodes = set(graph.keys())
   For each key, for each neighbor in graph[key], add neighbor to nodes.
2. Initialize in-degree dict for each node: in_deg = {node: 0 for node in nodes}
3. For each key, for each neighbor in graph[key], increment in_deg[neighbor] by 1.
4. Use a queue (collections.deque) for nodes with in_deg == 0.
5. Also maintain a list `result` for topological order.
6. While queue is not empty:
   pop node u, append to result.
   for each neighbor v of u (if u in graph, else empty list):
       decrement in_deg[v]
       if in_deg[v] == 0: queue.append(v)
7. After loop, if len(result) != len(nodes): there is a cycle.
   We need to raise ValueError with a message naming at least one node involved in the cycle.
   How to find a node in the cycle? Since we have in_deg remaining > 0 for some nodes, we can pick any node that is not in result. But we need to ensure it's part of a cycle. In a directed graph, any node with remaining positive in-degree after Kahn's is part of a cycle (or reachable from a cycle). Actually, in a directed graph, if there's a cycle, all nodes in the cycle will have in-degree at least 1 from within the cycle, but some nodes might have in-degree >0 from outside but still be part of a cycle. However, it's guaranteed that there exists at least one node in the cycle. We can find a node that is not in result, and then maybe trace back? But we need to name at least one node involved in the cycle. We could just pick any node that has in_deg > 0 after the algorithm, but is that always in a cycle? Consider a DAG with a node that has in-degree from a node that is not processed? Actually, if the graph has a cycle, the nodes in the cycle will never be processed because they depend on each other. Nodes that are not in the cycle but depend on cycle nodes will also have in-degree >0 because their prerequisites are not satisfied. But those nodes might not be "in the cycle" per se, but they are part of the strongly connected component or reachable from it. The problem says: "If the graph contains a cycle, raise ValueError with a message naming at least one node involved in the cycle." So we need to name a node that is actually part of a cycle. Simply picking a node with remaining in-degree might not be in the cycle; it could be a node that depends on the cycle but is not itself in the cycle. For example: a -> b -> c -> b (cycle b,c), and d -> b. Then d has in-degree 0? Actually d has no incoming edges, so it would be processed first, then b and c remain. b and c are in the cycle. But if we have a -> b -> c -> b, and also e -> c, e has in-degree 0, processed, then c remains, but c is in the cycle. So picking any node with remaining in-degree might work? Let's test: Suppose graph: a -> b, b -> c, c -> b (cycle b,c). Nodes: a, b, c. In-deg: a:0, b: from a and c => 2, c: from b => 1. Queue starts with a. Process a: decrement b's in-deg to 1. Queue empty. Remaining nodes: b and c, both have in_deg > 0. b is in cycle, c is in cycle. So picking b works. What if we have a -> b, b -> c, c -> d, d -> b (cycle b,c,d). Nodes: a,b,c,d. In-deg: a:0, b: from a and d => 2, c: from b => 1, d: from c => 1. Queue: a. Process a: b in-deg becomes 1. Queue empty. Remaining: b,c,d all have in_deg > 0. b is in cycle. So picking any remaining node works? What if there's a node that is not in the cycle but depends on a cycle node, and also has incoming edges from outside the cycle? Example: a -> b, b -> c, c -> b (cycle b,c). And also d -> c, and e -> d. Nodes: a,b,c,d,e. In-deg: a:0, b: from a and c => 2, c: from b and d => 2, d: from e => 1, e:0. Queue: a, e. Process a: b in-deg 1. Process e: d in-deg 0? Wait e has in-deg 0, process e: d in-deg becomes 0, queue d. Process d: c in-deg becomes 1. Queue empty. Remaining: b and c have in_deg 1 each. b and c are in the cycle. So again remaining nodes are in the cycle. Is it always true that any node with remaining in-degree after Kahn's is part of a cycle? Actually, in a directed graph, after removing all nodes with in-degree 0 iteratively, the remaining nodes form a subgraph where every node has in-degree at least 1 within that subgraph. This subgraph is the "core" and contains all cycles, but it may also include nodes that are not in any cycle but have edges only from within the core. However, if a node has in-degree >0 from within the core, it might still be part of a cycle or not. Consider a graph: a -> b, b -> c, c -> b (cycle b,c). And also d -> c, but d has no incoming edges from outside? Actually d has in-degree 0, so it gets processed. What if we have a node x that has an edge from a cycle node, but x itself has no outgoing edges to the cycle, and x has no other incoming edges? Then x would have in-degree 1 from the cycle, but after cycle nodes are not processed, x remains with in-degree 1. But x is not in the cycle; it's just a node that depends on the cycle. However, in Kahn's algorithm, if x has in-degree 1 from a cycle node, and that cycle node never gets processed, x will never have its in-degree reduced to 0. So x will remain in the "remaining" set. But x is not part of the cycle. So if we just pick any remaining node, we might pick x, which is not in the cycle. The problem requires naming at least one node involved in the cycle. So we need to ensure we name a node that is actually in a cycle.

Thus, Kahn's algorithm alone might not directly give a cycle node. We could use DFS-based topological sort with explicit cycle detection and extraction.

Alternative: Use DFS with three states: WHITE (unvisited), GRAY (in current recursion stack), BLACK (fully processed). When we encounter a GRAY node, we have found a back edge, indicating a cycle. At that point, we can extract the cycle by following the recursion stack from the current node back to the GRAY node. Since we need to name at least one node, we can just record the node where the back edge was found (the GRAY node) or the node we are currently visiting. But we must be careful: the back edge goes from current node u to v, where v is GRAY. Then v is in the cycle. So we can raise ValueError with message involving v.

But we also need to include all nodes in the output, even those that appear only as targets. In DFS topological sort, we typically visit all nodes. We need to ensure we visit all nodes, including those not in the graph keys. So we'll have a set of all nodes, and we iterate over them, starting DFS if not visited.

DFS topological sort algorithm:

- Initialize visited = set() (or dict with states: 0=unvisited, 1=in progress, 2=done)
- result = []
- For each node in all_nodes:
    if node not in visited (or state 0):
        dfs(node)
- dfs(u):
    mark u as GRAY (state 1)
    for each v in graph.get(u, []):
        if state[v] == 1:  # cycle detected
            # we can raise ValueError immediately, but we need to name a node in cycle.
            # v is in the cycle. We can raise ValueError(f"Graph contains a cycle involving node {v}")
            # But we must ensure we don't continue processing after cycle detection? We can just raise.
        elif state[v] == 0:
            dfs(v)
    mark u as BLACK (state 2)
    append u to result

But note: if we raise ValueError inside dfs, we need to propagate it. Also, we need to handle the case where the graph has a cycle but we might have already visited some nodes. The error message should name at least one node involved in the cycle. v is definitely in the cycle because it's an ancestor in the current DFS path. So we can just raise ValueError with that node.

But we must be careful: the graph might have multiple components, and we might detect a cycle in one component. That's fine.

However, we need to ensure that we include all nodes in the output even if there's no cycle. The DFS approach naturally produces a topological order (reverse of finishing times). But we must include all nodes, including those that are only targets. We'll collect all nodes first.

Let's test the DFS approach on the example: topo_sort({"a": ["b"], "b": ["c"], "c": []}) => ["a", "b", "c"].

All nodes: a, b, c. Start DFS from a: visit a (GRAY), neighbors: b (state 0) -> dfs(b): visit b (GRAY), neighbors: c (state 0) -> dfs(c): visit c (GRAY), neighbors: none -> mark c BLACK, append c to result. Return to b: mark b BLACK, append b. Return to a: mark a BLACK, append a. Result: [c, b, a]? Wait, we append after processing neighbors, so the order is reverse of finishing times. Typically topological sort using DFS appends node after all its descendants are processed, so the result list is in reverse topological order? Actually, standard Kahn's gives topological order where dependencies come first. DFS-based topological sort: if we append node after visiting all its neighbors, the resulting list is a topological order (i.e., for every directed edge u -> v, u appears before v in the list). Let's check: In the example, if we start from a, we get result = [c, b, a] if we append after processing. But we want ["a", "b", "c"]. So we can either reverse the result, or prepend. Usually, we append to a list and then reverse at the end, or we can insert at front. But the problem says: "Return a list of all nodes in an order such that every node appears before the nodes it points to." So if a points to b, a must appear before b. In the example, a -> b -> c, so a before b before c. If we do DFS and append after processing, we get c, b, a. Reversing gives a, b, c. So we can just reverse the result list at the end.

But we must be careful: if we detect a cycle, we raise ValueError before finishing. So we can just do the DFS, and if we find a back edge, we raise ValueError with the node v (the GRAY node). But we need to ensure that v is indeed in the cycle. Yes, because it's on the current recursion stack.

But what if the graph has a cycle but we start DFS from a node not in the cycle? We will eventually visit all nodes, and when we encounter a back edge, we raise. That's fine.

Now, we need to handle nodes that appear only as targets. We'll collect all nodes from keys and values. Then we'll initialize state dict for all nodes: state = {node: 0 for node in all_nodes}. Then for each node in all_nodes, if state[node] == 0, we call dfs(node). In dfs, we need to iterate over graph.get(node, []) to get neighbors. But note: some nodes might not have outgoing edges, so graph.get(node, []) returns [].

One edge case: empty graph. If graph is {}, then all_nodes will be empty set. We should return []. The algorithm should handle that.

Another edge case: graph with nodes that have no outgoing edges and are not keys, e.g., {"a": ["b"]}. Then all_nodes = {"a", "b"}. DFS from a will visit a and b. Result after reverse will be ["a", "b"]? Let's trace: all_nodes = {"a", "b"}. state: a:0, b:0. Start with a (or maybe iterate over set, order not guaranteed, but we can sort or just use any order; topological order may vary but must be valid). Suppose we start with a: dfs(a): mark a GRAY, neighbors: b (state 0) -> dfs(b): mark b GRAY, neighbors: graph.get(b, []) -> [] (since b not in graph). mark b BLACK, append b. Return to a: mark a BLACK, append a. Result before reverse: [b, a]. Reverse: [a, b]. That's correct: a before b.

What if we start with b first? Since we iterate over all_nodes, which is a set, order might be arbitrary. If we start with b: dfs(b): mark b GRAY, neighbors: [], mark b BLACK, append b. Then dfs(a): mark a GRAY, neighbors: b (state 2? Actually b is already visited and BLACK, so state[b] == 2, so we don't recurse, just skip). Then mark a BLACK, append a. Result before reverse: [b, a] again? Wait, if we start with b, we append b, then process a, append a. Result [b, a], reverse [a, b]. Still correct. But what if there's a node that is only a target and we start with it, and there's a cycle elsewhere? We'll detect cycle when we visit the cycle node. But if we start with a target node that has no outgoing edges, it will be processed first, then we process other nodes. That's fine.

But we must ensure that we include all nodes in the output even if there's a cycle? The requirement says: "If the graph contains a cycle, raise ValueError with a message naming at least one node involved in the cycle." So we don't need to return anything if there's a cycle; we just raise.

Now, about the error message: "naming at least one node involved in the cycle". We can just say "Graph contains a cycle involving node {v}". But we need to be sure v is in the cycle. In DFS, when we are at node u and we see neighbor v with state 1, v is an ancestor in the current DFS path, so v is definitely in a cycle. However, the cycle might include more nodes, but v is one of them. So that's acceptable.

But what if the graph has a self-loop? e.g., {"a": ["a"]}. Then all_nodes = {"a"}. DFS from a: mark a GRAY, neighbor a has state 1 -> cycle detected. v = a. Raise ValueError("Graph contains a cycle involving node a"). That's correct.

What about a graph with multiple cycles? We'll raise on first back edge encountered.

Now, we need to include type hints. The function signature: `topo_sort(graph: dict) -> list`. But we need to specify the types of keys and values. The graph is a dict mapping each node to a list of the nodes it points to. So we can use `dict[Any, list[Any]]` or more specifically `dict[T, list[T]]` but we don't have a type variable. We can use `dict` and `list` without generic, or use `TypeVar`. The requirement says "Include type hints". We can do:

from typing import Any, Dict, List

def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:

But nodes could be strings, ints, etc. Using `Any` is fine.

Alternatively, we can use `dict` and `list` without generic, but type hints usually expect something. We'll use `Any`.

Now, let's write the code.

Pseudo-code:

def topo_sort(graph: dict) -> list:
    # Collect all nodes
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    
    if not all_nodes:
        return []
    
    # State: 0 = unvisited, 1 = in progress (gray), 2 = done (black)
    state = {node: 0 for node in all_nodes}
    result = []
    
    def dfs(u: Any) -> None:
        state[u] = 1  # gray
        for v in graph.get(u, []):
            if state[v] == 1:
                # cycle detected
                raise ValueError(f"Graph contains a cycle involving node {v}")
            elif state[v] == 0:
                dfs(v)
        state[u] = 2  # black
        result.append(u)
    
    for node in all_nodes:
        if state[node] == 0:
            dfs(node)
    
    # Reverse result to get topological order
    return result[::-1]

Wait, we need to ensure that the error message names at least one node involved in the cycle. The node v is indeed in the cycle. But what if the graph has a cycle but we start DFS from a node that is not in the cycle, and we encounter a back edge to a node that is in the cycle? That's fine, v is in the cycle.

But there's a subtlety: In DFS, when we mark a node as gray, and we encounter a neighbor that is gray, that neighbor is an ancestor in the current DFS path. However, it's possible that the gray node is not part of a cycle if the graph has a forward edge? No, in a directed graph, a back edge to a gray node always indicates a cycle. Because the gray node is currently being processed, and we are trying to visit it again, meaning there's a path from the current node back to it. So it's a cycle.

But we must be careful: The graph might have nodes that are not reachable from the start, but we iterate over all_nodes. That's fine.

Now, test with example: topo_sort({"a": ["b"], "b": ["c"], "c": []}) => all_nodes = {"a","b","c"}. Suppose we start with "a": dfs("a"): state a=1, neighbors: b (state 0) -> dfs(b): state b=1, neighbors: c (state 0) -> dfs(c): state c=1, neighbors: [] -> state c=2, result.append(c). Return to b: state b=2, result.append(b). Return to a: state a=2, result.append(a). result = [c, b, a]. Reverse: [a, b, c]. Correct.

Test with cycle: {"a": ["b"], "b": ["a"]}. all_nodes = {"a","b"}. Suppose start with a: dfs(a): state a=1, neighbors: b (state 0) -> dfs(b): state b=1, neighbors: a (state 1) -> raise ValueError("Graph contains a cycle involving node a"). Correct.

What if we start with b first? Then raise involving b. Both are in the cycle.

What if graph has a cycle but also other nodes? {"a": ["b"], "b": ["c"], "c": ["b"]}. all_nodes = {"a","b","c"}. Suppose start with a: dfs(a): state a=1, neighbors: b (state 0) -> dfs(b): state b=1, neighbors: c (state 0) -> dfs(c): state c=1, neighbors: b (state 1) -> raise ValueError("Graph contains a cycle involving node b"). b is in the cycle. Correct.

What if the cycle is not reachable from the first node we pick? We iterate over all_nodes, so we will eventually start DFS from a node in the cycle, or from a node that leads to the cycle. When we encounter the back edge, we raise. So it's fine.

Now, what about nodes that appear only as targets and have no outgoing edges, and we start DFS from them? They will be processed, state set to 2, appended. Then later when we process other nodes, they might have edges to already visited nodes (state 2), which we skip. That's fine.

Now, check the requirement: "Nodes that appear only as targets (never as keys) must still be included in the output". Our algorithm collects all nodes from keys and values, so they will be included.

"Handle the empty graph": if graph is {}, all_nodes will be empty set, we return [].

"Include type hints": we'll add them.

"O(V + E)": DFS visits each node once and each edge once. Collecting all nodes: we iterate over keys and values, which is O(V + E) as well. So overall O(V + E).

One potential issue: In the DFS, we use `graph.get(u, [])`. But what if a node is in all_nodes but not in graph keys? Then graph.get returns []. That's correct.

But we must ensure that we don't miss any edges. The graph dict maps each node to a list of nodes it points to. So for nodes that are keys, we have their outgoing edges. For nodes that are not keys, they have no outgoing edges (or maybe they are not in the graph at all, but we added them as targets, so they have no outgoing edges specified). That's correct per the problem: "graph is a dict mapping each node to a list of the nodes it points to (its dependencies-after, i.e. edges go from key to each element)." So if a node is only a target, it has no outgoing edges in the given dict. So graph.get(node, []) is correct.

Now, consider the case where a node appears as a target and also as a key? That's fine.

Now, what about the error message format? The problem says: "raise ValueError with a message naming at least one node involved in the cycle." It doesn't specify exact format, but we should include a node name. Our message "Graph contains a cycle involving node {v}" seems appropriate. But we need to ensure that v is indeed a node in the cycle. As argued, yes.

But there's a corner case: What if the graph has a cycle, but the back edge we detect is from u to v where v is gray, but v might not be "involved in the cycle" if the cycle is actually from u to some other node? No, if v is gray, it's on the current DFS stack, meaning there's a path from the start node to v, and now we are at u and we see an edge to v, creating a cycle that includes v and u and the path between them. So v is definitely in the cycle.

But what if the graph has a self-loop? v = u, and state[u] == 1, we raise involving u. Correct.

Now, is there any case where we might raise ValueError but the node we name is not actually in a cycle? I don't think so.

But wait: In DFS, when we mark a node as gray, and we iterate over its neighbors, if we encounter a neighbor that is gray, that neighbor is an ancestor. However, it's possible that the gray node is not part of a cycle if the graph has a cross edge? In directed graphs, a cross edge goes between different branches and doesn't create a cycle. But a cross edge to a gray node? Actually, in standard DFS on directed graphs, we have three types of edges: tree edges, back edges (to ancestor -> cycle), forward edges (to descendant), and cross edges (to unrelated node). A cross edge cannot go to a gray node because gray nodes are only on the current path. If we encounter a gray node, it must be an ancestor, hence a back edge. So it's always a cycle.

But there's a nuance: In our DFS, we are using a global state dict. When we start a new DFS from a node that is already visited (state 2), we skip it. When we start from a node with state 0, we go deep. The gray state is only set for the current recursion stack. So if we encounter a gray node, it's definitely a back edge.

One potential issue: What if the graph has a cycle, but we start DFS from a node that is not in the cycle, and we visit nodes that are not in the cycle, and then we encounter a back edge to a gray node that is in the cycle? That's fine, we name that gray node.

But what if the graph has a cycle, but we start DFS from a node that is in the cycle, and we might detect the cycle immediately? That's fine.

Now, let's test with a more complex graph: {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}. all_nodes = {"a","b","c","d"}. DFS from a: a->b, b->d, d done, b done, a->c, c->d (d already state 2, skip), c done, a done. Result before reverse: [d, b, c, a]? Let's trace: start a: state a=1. neighbors b, c. Suppose we go b first: dfs(b): state b=1, neighbors d: dfs(d): state d=1, neighbors [] -> state d=2, result.append(d). Return to b: state b=2, result.append(b). Return to a: now go c: dfs(c): state c=1, neighbors d: state d=2, skip. state c=2, result.append(c). Return to a: state a=2, result.append(a). result = [d, b, c, a]. Reverse: [a, c, b, d]? Wait, reverse of [d, b, c, a] is [a, c, b, d]. But is that a valid topological order? a before c, a before b, b before d, c before d. In [a, c, b, d], a is before c and b, c before d, b before d. But is b before d? Yes, b is at index 2, d at index 3. c is at index 1, d at index 3. So a before b and c, b before d, c before d. That's valid. But note that the order a, c, b, d has c before b, which is fine because there's no edge between b and c. So it's a valid topological sort.

Now, what if we have a graph where a node has multiple incoming edges and we need to ensure all nodes are included. Our algorithm does that.

Now, consider the requirement: "The algorithm must be O(V + E)". Our algorithm: collecting all nodes: we iterate over graph.keys() and graph.values(). graph.values() gives lists; we iterate over each neighbor. That's O(V + E) because sum of lengths of adjacency lists is E. Then we have a loop over all_nodes, and inside dfs we visit each node once and each edge once. So total O(V + E).

Now, type hints: We'll use `from typing import Any, Dict, List`. But the problem says "Include type hints". We can just put `def topo_sort(graph: dict) -> list:` but that's not very specific. Better to use `Dict` and `List` from typing. However, the problem might accept `dict` and `list`. I'll use `Dict[Any, List[Any]]` and `List[Any]`. But we need to import `Any`, `Dict`, `List`. I'll do:

from typing import Any, Dict, List

def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:

But wait: The graph might have nodes that are not hashable? No, nodes are typically hashable (strings, ints). Using `Any` is fine.

Alternatively, we can not use generic types and just use `dict` and `list`. The requirement says "Include type hints", so we should include them. I'll use `Dict` and `List` with `Any`.

Now, let's think about edge cases:

1. Empty graph: `topo_sort({})` -> all_nodes = set(), return [].

2. Graph with one node and no edges: `{"a": []}` -> all_nodes = {"a"}, DFS from a: state a=1, neighbors [] -> state a=2, result.append(a). Reverse: [a]. Correct.

3. Graph with one node pointing to itself: `{"a": ["a"]}` -> all_nodes = {"a"}, DFS from a: state a=1, neighbor a state 1 -> raise ValueError("Graph contains a cycle involving node a").

4. Graph with nodes only as targets: `{"a": ["b"]}` -> all_nodes = {"a","b"}, DFS from a: a->b, b has no outgoing edges. Result before reverse: [b, a] -> reverse [a, b]. Correct.

5. Graph with multiple disconnected components: `{"a": ["b"], "c": []}` -> all_nodes = {"a","b","c"}. Suppose we start with a: a->b, b done, a done. Then start with c: c done. Result before reverse: [b, a, c]? Let's trace: start a: state a=1, neighbors b (state 0) -> dfs(b): state b=1, neighbors [] -> state b=2, result.append(b). Return to a: state a=2, result.append(a). result = [b, a]. Then iterate over all_nodes: next node c (state 0) -> dfs(c): state c=1, neighbors [] -> state c=2, result.append(c). result = [b, a, c]. Reverse: [c, a, b]. Is that valid? a before b, c has no edges. So c, a, b is valid. But what if we started with c first? Then result might be [a, b, c] or [c, a, b] depending on iteration order. Both are valid topological orders.

6. Graph with a cycle and other nodes: `{"a": ["b"], "b": ["a"], "c": []}`. all_nodes = {"a","b","c"}. Suppose we start with a: dfs(a): state a=1, neighbors b (state 0) -> dfs(b): state b=1, neighbors a (state 1) -> raise ValueError("Graph contains a cycle involving node a"). Correct.

What if we start with c first? Then we process c, state c=2, result.append(c). Then we start with a: same cycle detection. So we raise.

Now, is there any case where we might not detect a cycle because we skip nodes that are already visited? No, because we only skip if state == 2. If a node is in a cycle, it will never reach state 2 because it will keep being visited and we will eventually hit a back edge. But what if the cycle is only reachable through a node that we already visited and marked as 2? That's impossible because if a node is in a cycle, it has outgoing edges to other cycle nodes, and those have edges back. If we start DFS from a node that is not in the cycle, we will eventually enter the cycle and then detect the back edge. If we start from a node in the cycle, we detect immediately. So we will always detect a cycle if one exists.

But there's a subtle case: What if the graph has a cycle, but all nodes in the cycle have in-degree from outside the cycle, and we start DFS from a node outside, and we visit all outside nodes, and then we start from a node in the cycle? We iterate over all_nodes, so we will eventually start from a node in the cycle. But what if the cycle nodes are only reachable from outside, and we process outside first, marking some nodes as 2, but the cycle nodes are not yet visited. Then we will later start from a cycle node (since it's in all_nodes and state 0). That's fine.

Now, what about the error message: "naming at least one node involved in the cycle". Our message names v, which is the node we are trying to visit that is already gray. Is it guaranteed that v is "involved in the cycle"? Yes, because the cycle consists of the path from v to u (where u is current node) plus the edge u->v. So v is in the cycle. However, what if the graph has a self-loop? v = u, and u is in the cycle (the self-loop). So that's fine.

But consider a graph where we have a back edge to a gray node, but that gray node is not part of a cycle because the graph has a forward edge? No, as discussed, back edge to gray always means cycle.

One potential issue: In our DFS, we set state[u] = 1 at the beginning, and then we iterate neighbors. If we encounter a neighbor v with state 1, we raise. But what if the graph has a node that is gray because it's an ancestor, but the edge u->v is not a back edge but a cross edge? In directed DFS, a cross edge goes to a node that is already finished (state 2) or to a node in a different branch that is currently gray? Actually, in standard DFS on directed graphs, when we are at node u and we look at neighbor v, if v is gray, it must be an ancestor in the current DFS tree, because all other nodes are either white (unvisited) or black (finished). There is no other state. So it's always a back edge. So it's safe.

But wait: What if we have a graph where a node has multiple parents, and we visit it via one parent, mark it gray, then finish it, mark it black. Then later we visit it via another parent? We only visit a node if its state is 0. Once it's black (state 2), we skip it. So we won't encounter a gray node that was already finished. So the only way to encounter a gray node is if it's currently on the recursion stack, i.e., an ancestor. So it's a back edge.

Thus, the algorithm is correct.

Now, let's write the final code.

We need to make sure we import `Any`, `Dict`, `List`. We'll put the function definition.

One more thing: The problem says "Signature: `topo_sort(graph)` where `graph` is a dict mapping each node to a list of the nodes it points to (its dependencies-after, i.e. edges go from key to each element)." So the graph is a dict. We'll use that.

We'll also handle the empty graph: if graph is empty dict, return [].

But what if graph is not empty but has no keys? That's empty dict.

Now, let's write the code carefully.

```python
from typing import Any, Dict, List

def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    # Collect all nodes: keys and values
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    
    if not all_nodes:
        return []
    
    # State: 0 = unvisited, 1 = in progress (gray), 2 = done (black)
    state = {node: 0 for node in all_nodes}
    result = []
    
    def dfs(u: Any) -> None:
        state[u] = 1  # gray
        for v in graph.get(u, []):
            if state[v] == 1:
                # Cycle detected; v is involved in the cycle
                raise ValueError(f"Graph contains a cycle involving node {v}")
            elif state[v] == 0:
                dfs(v)
        state[u] = 2  # black
        result.append(u)
    
    for node in all_nodes:
        if state[node] == 0:
            dfs(node)
    
    # Reverse to get topological order (dependencies before dependents)
    return result[::-1]
```

Wait, there's a potential issue: In the DFS, we iterate over `graph.get(u, [])`. But what if a node is in `all_nodes` but not in `graph`, we get `[]`. That's fine. But what if a node is in `graph` but its value list contains a node that is not in `all_nodes`? We already added all values to `all_nodes` in the collection step, so every neighbor is in `all_nodes`. So `state[v]` will be defined.

But there's a subtle bug: In the collection step, we do `all_nodes.update(neighbors)` for each `neighbors` in `graph.values()`. But what if a neighbor is itself a key in the graph? That's fine, it's already in the set. But what if a neighbor is a node that is not hashable? We assume nodes are hashable (strings, ints). The problem doesn't specify, but topological sort typically uses hashable nodes.

Now, test with the example: `topo_sort({"a": ["b"], "b": ["c"], "c": []})` -> returns `["a", "b", "c"]`. Let's mentally run: all_nodes = {"a","b","c"}. state: a:0, b:0, c:0. Iterate over all_nodes (order might vary, but suppose we start with "a"). dfs("a"): state a=1, neighbors: ["b"] -> v="b", state b=0 -> dfs("b"): state b=1, neighbors: ["c"] -> v="c", state c=0 -> dfs("c"): state c=1, neighbors: [] -> state c=2, result.append("c"). Return to b: state b=2, result.append("b"). Return to a: state a=2, result.append("a"). result = ["c", "b", "a"]. Reverse: ["a", "b", "c"]. Correct.

What if we start with "c" first? all_nodes iteration: suppose we start with "c": dfs("c"): state c=1, neighbors [] -> state c=2, result.append("c"). result = ["c"]. Then next node "a": dfs("a"): state a=1, neighbors ["b"] -> dfs("b"): state b=1, neighbors ["c"] -> state c=2 (already done), so we skip (elif state[v] == 0 is false, state[v] == 1 is false, so we do nothing? Wait, in our code, we have:

```python
        for v in graph.get(u, []):
            if state[v] == 1:
                raise ValueError(...)
            elif state[v] == 0:
                dfs(v)
```

If state[v] == 2, we do nothing (just skip). That's correct. So in dfs("b"), neighbor "c" has state 2, so we skip. Then state b=2, result.append("b"). Return to a: state a=2, result.append("a"). result = ["c", "b", "a"]. Reverse: ["a", "b", "c"]. Then next node "b" in all_nodes iteration: state b is already 2, so we skip. So final result ["a", "b", "c"]. Correct.

Now, what if we have a graph with a cycle and we start with a node not in the cycle? We already tested.

Now, consider the error message: "Graph contains a cycle involving node {v}". The problem says "raise ValueError with a message naming at least one node involved in the cycle." Our message names v. But what if v is not actually in the cycle? As argued, it is. But let's double-check with a tricky case: Suppose we have a graph with a forward edge? In directed graphs, a forward edge is from an ancestor to a descendant, but the descendant is already finished (black). So we wouldn't encounter a gray node. A cross edge goes to a node that is already finished or in a different branch, but not gray. So no issue.

But what if the graph has a node that is gray because we are in the middle of DFS, and we see an edge to a gray node that is not an ancestor? In a standard DFS, that's impossible because gray nodes are exactly those on the current path from the root. Any other node is either white or black. So it's always an ancestor.

Thus, the algorithm is correct.

Now, one more thing: The problem says "If the graph contains a cycle, raise ValueError with a message naming at least one node involved in the cycle." It doesn't specify the exact format, but we should make sure the message is clear. Our message "Graph contains a cycle involving node {v}" seems fine.

But what if the cycle involves multiple nodes, and we name one, that's acceptable.

Now, let's consider if there's any case where we might raise ValueError but the node we name is not actually in the cycle due to some DFS ordering artifact? I think it's solid.

Now, check the requirement "Handle the empty graph". Our code: if not all_nodes: return []. If graph is {}, all_nodes is empty set, returns []. If graph has keys but no values? e.g., {"a": []} -> all_nodes = {"a"}, not empty, returns ["a"]. Good.

Now, type hints: We have `from typing import Any, Dict, List`. The function signature: `def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:`. This is good.

But wait: The problem says "Signature: `topo_sort(graph)` where `graph` is a dict mapping each node to a list of the nodes it points to (its dependencies-after, i.e. edges go from key to each element)." It doesn't specify the types of nodes, so `Any` is appropriate.

Now, is there any requirement about the order of nodes in the output when there are multiple valid topological orders? No, any valid order is fine.

Now, let's think about potential improvements or edge cases:

- What if the graph has nodes that are not strings but integers, or tuples? `Any` handles it.
- What if the graph has a node that appears as a target and also as a key, but with different outgoing edges? That's fine.
- What if the graph has a node that is only a target, and we start DFS from it, and it has no outgoing edges, it gets processed. Then later we process other nodes that might have edges to it. Since it's already state 2, we skip. That's correct.
- What if the graph has a cycle that is not reachable from the first node we pick, but we iterate over all_nodes, so we will eventually start from a node in the cycle. But what if the cycle nodes are all in a component that we never start from because we skip them? We iterate over all_nodes, and for each node, if state[node] == 0, we dfs. So we will eventually start from a cycle node if it's unvisited. But what if all cycle nodes are already visited (state 2) before we get to them? That can't happen because if they are in a cycle, they can never reach state 2 without the DFS completing, which would require breaking the cycle. But if we start from a node outside the cycle, we might visit some cycle nodes and then hit a back edge and raise. If we start from outside and somehow visit all cycle nodes without hitting a back edge? That's impossible because to visit a cycle node, we must follow edges from outside into the cycle. Once we enter the cycle, we will follow edges and eventually try to visit a node already on the stack, causing a back edge. So we will raise before marking all cycle nodes as 2. So we will always detect the cycle.

But there's a corner case: What if the graph has a cycle, but all nodes in the cycle have in-degree from outside the cycle, and we start DFS from a node outside, and we visit all outside nodes, and then we start from a cycle node? We iterate over all_nodes, so we will eventually start from a cycle node. But what if the cycle nodes are already visited because we started from another outside node? We only visit a node if state is 0. If we start from outside, we might visit some cycle nodes and raise. If we don't raise (impossible), we would mark them as 2. But we will raise.

Now, what if the graph has a cycle, but we start DFS from a node that is in the cycle, and the first neighbor we check is the one that creates the back edge? We raise immediately. Good.

Now, let's test with a graph that has a cycle but also a node that is only a target and has no edges: `{"a": ["b"], "b": ["a"], "c": []}`. all_nodes = {"a","b","c"}. Suppose we start with "c": dfs("c"): state c=1, neighbors [] -> state c=2, result.append("c"). result = ["c"]. Then iterate over all_nodes: next "a": dfs("a"): state a=1, neighbors ["b"] -> dfs("b"): state b=1, neighbors ["a"] -> state a=1 -> raise ValueError("Graph contains a cycle involving node a"). Correct.

What if we start with "a" first? Same raise.

Now, what if the graph has a cycle but the cycle is only reachable through a node that we process after the cycle? We iterate over all_nodes, so order doesn't matter; we will eventually start from a cycle node or from a node that leads to the cycle. If we start from a node that leads to the cycle, we will enter the cycle and detect back edge. If we start from a cycle node, we detect immediately. So we always detect.

Now, is there any case where we might not detect a cycle because we skip nodes that are already visited (state 2) and the cycle is somehow "broken" by previous visits? No, because we only skip if state is 2, and we only set state 2 after fully processing a node and its descendants. If a node is in a cycle, it will never be fully processed without encountering a back edge, because to process it, we must follow its outgoing edges, which lead back to itself or other cycle nodes, creating a loop. The only way to process a cycle node is if we somehow break the cycle by not following some edge, but we follow all edges. So we will always detect.

Thus, the algorithm is sound.

Now, let's consider the time complexity: Collecting all nodes: we iterate over graph.keys() and graph.values(). graph.values() gives lists; we iterate over each neighbor. That's O(V + E) because sum of lengths of adjacency lists is E. Then we have a loop over all_nodes, and inside dfs we visit each node once and each edge once. The for loop over all_nodes might call dfs on each node, but each node is visited only once because state prevents re-visiting. So total O(V + E).

Space complexity: state dict O(V), result list O(V), recursion stack O(V) in worst case (depth of DFS). So O(V + E) time, O(V) space.

Now, we should also consider if the graph is very large, recursion depth might exceed Python's recursion limit. The problem doesn't specify we must use iterative DFS, but recursive DFS is simpler and O(V+E). However, to be safe and meet O(V+E) without recursion limit issues, we could use an iterative approach, but the problem doesn't explicitly forbid recursion. Usually, in such problems, recursive DFS is acceptable, but we might want to mention that it uses recursion. The requirements say "The algorithm must be O(V + E)". Recursive DFS is O(V+E) time, but space O(V) for call stack. That's fine. We could also implement Kahn's algorithm iteratively, which is also O(V+E) and avoids recursion depth issues, but we need to detect cycle and name a node. Kahn's algorithm is easier for cycle detection? As discussed, Kahn's doesn't directly give a cycle node. But we could modify it. However, the DFS approach is straightforward and naturally names a cycle node. I'll stick with recursive DFS, but I should note that for very deep graphs, recursion might hit limit. But the problem doesn't mention recursion limit, so it's fine.

Alternatively, we could implement Kahn's algorithm and then, if there's a cycle, find a node in the cycle by looking at the remaining nodes and maybe doing a DFS on the remaining subgraph to find a cycle. But that would be more complex. The DFS approach is simpler and meets all requirements.

Let's double-check the error message format. The problem says: "raise ValueError with a message naming at least one node involved in the cycle." Our message: `f"Graph contains a cycle involving node {v}"`. This names node v. Is v always a node? Yes, v is a node from the graph. But what if the graph has nodes that are not strings? v could be an int, etc. The f-string will convert it to string. That's fine.

But wait: In the DFS, when we detect a back edge, we are at node u, and we see neighbor v with state 1. We raise with v. But what if the cycle is actually from u to v, but v is not "involved in the cycle" because the cycle might be something else? No, v is definitely in the cycle. However, consider a graph where we have a node that is gray, but the edge u->v is not part of a cycle because v is an ancestor but the path from v to u doesn't include v? That's contradictory: if v is an ancestor, there is a path from the start node to v, and then from v we went to some nodes, and now we are at u, and we have an edge u->v. That forms a cycle that includes v and u and the path between them. So v is in the cycle.

But what if the graph has a node that is gray because we are in a different branch? In DFS, gray nodes are only those on the current path from the root. There are no other gray nodes. So it's always an ancestor.

Thus, the algorithm is correct.

Now, let's write the final answer with the code inside `<answer>` tags.

But before finalizing, let's think if there's any edge case where our DFS might not visit all nodes because we start from a node that is not in the graph keys but is a target, and we might miss some nodes? We iterate over `all_nodes`, which includes all keys and values. So we will start DFS from every node that has state 0. Since we initialize state for all nodes, we will eventually visit all nodes. The only issue is if the graph has a node that is not reachable from any other node, but we start from it because it's in `all_nodes` and state 0. So we will visit it.

What if the graph has a node that is a target and also a key, but we start from it, and it has outgoing edges. That's fine.

Now, one more thing: The problem says "Nodes that appear only as targets (never as keys) must still be included in the output". Our algorithm includes them because we add them to `all_nodes`. But what if a node appears only as a target, and we start DFS from it, and it has no outgoing edges, it gets appended. Then later, when we process other nodes that have edges to it, we skip it because it's already state 2. That's correct; it's included in the output.

But what if a node appears only as a target, and we start DFS from some other node that has an edge to it, we will visit it and it will be included. So all good.

Now, let's test with a graph where a node is only a target and there's a cycle elsewhere: `{"a": ["b"], "b": ["a"], "c": []}`. We already tested, c is included in output if no cycle? But there is a cycle, so we raise ValueError. If we remove the cycle: `{"a": ["b"], "c": []}` -> all_nodes = {"a","b","c"}. Suppose we start with a: a->b, b done, a done. Then start with c: c done. Result before reverse: [b, a, c]? Let's trace: start a: state a=1, neighbors b (state 0) -> dfs(b): state b=1, neighbors [] -> state b=2, result.append(b). Return to a: state a=2, result.append(a). result = [b, a]. Then start c: state c=1, neighbors [] -> state c=2, result.append(c). result = [b, a, c]. Reverse: [c, a, b]. Output: ["c", "a", "b"]. That includes c. Correct.

What if we start with c first? Then result before reverse: [c, b, a]? Let's see: start c: state c=1, neighbors [] -> state c=2, result.append(c). result = [c]. Then start a: state a=1, neighbors b (state 0) -> dfs(b): state b=1, neighbors [] -> state b=2, result.append(b). Return to a: state a=2, result.append(a). result = [c, b, a]. Reverse: [a, b, c]. Output: ["a", "b", "c"]. Includes c. Correct.

Now, what if the graph has a node that is only a target and also has outgoing edges? The problem says "graph is a dict mapping each node to a list of the nodes it points to". If a node appears only as a target, it might not be a key, so it has no outgoing edges in the given dict. But what if a node appears as a target and also as a key? That's allowed; it will have outgoing edges from its key entry. Our algorithm handles that because we add all keys and values to all_nodes, and for keys we use graph.get(node, []). If a node is a key, we get its list; if it's only a target, we get [].

Now, is there any case where a node appears as a target but also has outgoing edges that are not in the dict? No, the graph only contains the edges specified.

Now, let's consider the type hints. The problem says "Include type hints". We'll include them.

One potential improvement: We could use `from __future__ import annotations` but not needed.

Now, final code:

```python
from typing import Any, Dict, List

def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    # Collect all nodes: keys and values (targets)
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    
    if not all_nodes:
        return []
    
    # State: 0 = unvisited, 1 = in progress (gray), 2 = done (black)
    state = {node: 0 for node in all_nodes}
    result = []
    
    def dfs(u: Any) -> None:
        state[u] = 1  # gray
        for v in graph.get(u, []):
            if state[v] == 1:
                # Cycle detected; v is involved in the cycle
                raise ValueError(f"Graph contains a cycle involving node {v}")
            elif state[v] == 0:
                dfs(v)
        state[u] = 2  # black
        result.append(u)
    
    for node in all_nodes:
        if state[node] == 0:
            dfs(node)
    
    # Reverse to obtain topological order (dependencies before dependents)
    return result[::-1]
```

Wait, there's a small issue: In the for loop `for node in all_nodes:`, the order of iteration over a set is arbitrary. That's fine for topological sort; any valid order is acceptable. But we must ensure that we don't miss any nodes. Since we check `if state[node] == 0`, we will eventually visit all nodes.

But what if the graph has a node that is only a target and we start from it, and it has no outgoing edges, it gets processed. Then later we start from another node that has an edge to it, we skip it. That's fine.

Now, test with a graph that has a cycle but the back edge is to a node that is not the one we name? We name v. Is it possible that v is not in the cycle? Let's construct a scenario: Suppose we have nodes a, b, c, d. Edges: a -> b, b -> c, c -> d, d -> b (cycle b,c,d). Also a -> e, e -> something? Let's just have a -> b, b -> c, c -> b (cycle b,c). And also d -> c, but d has no incoming edges? Actually, if we have a -> b, b -> c, c -> b, and also d -> c, and e -> d. Nodes: a,b,c,d,e. Suppose we start DFS from a: a->b, b->c, c->b (back edge to b). We raise involving b. b is in the cycle. Good.

What if we start from e: e->d, d->c, c->b, b->c? Wait, if we have e->d, d->c, c->b, b->c. That's a cycle c,b. If we start from e: e->d (state d=1), d->c (state c=1), c->b (state b=1), b->c (state c=1) -> raise involving c. c is in the cycle. Good.

What if we have a graph where a back edge is detected to a node that is gray, but that node is not actually in a cycle because the graph has a forward edge? As discussed, impossible.

But there's one edge case: In DFS, we set state[u] = 1 at the start, then we iterate neighbors. If we encounter a neighbor v with state 1, we raise. But what if the graph has a node that is gray because it's an ancestor, but the edge u->v is not a back edge but a cross edge to a node that is gray in a different branch? In standard DFS on directed graphs, when we are at node u, the set of gray nodes is exactly the ancestors of u in the current DFS tree. There are no other gray nodes. This is because we only set state to 1 when we enter a node, and set to 2 when we leave. We never set state to 1 for nodes not on the current path. So any gray node we encounter must be an ancestor. So it's a back edge.

Thus, the algorithm is correct.

Now, I'll produce the final answer.

</details>

**Answer (as scored):**

from typing import Any, Dict, List


def topo_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    # Collect all nodes: keys and values (targets that may never appear as keys)
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # Handle empty graph
    if not all_nodes:
        return []

    # State: 0 = unvisited, 1 = in progress (gray), 2 = done (black)
    state = {node: 0 for node in all_nodes}
    result = []

    def dfs(u: Any) -> None:
        state[u] = 1  # gray
        for v in graph.get(u, []):
            if state[v] == 1:
                # Cycle detected; v is involved in the cycle
                raise ValueError(f"Graph contains a cycle involving node {v}")
            elif state[v] == 0:
                dfs(v)
        state[u] = 2  # black
        result.append(u)

    for node in all_nodes:
        if state[node] == 0:
            dfs(node)

    # Reverse to obtain topological order (every node appears before the nodes it points to)
    return result[::-1]

