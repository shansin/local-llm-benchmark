"""Reference tests for the topo_sort task.

The valid answer is not unique, so these check the ordering property rather
than one specific permutation.
"""

import pytest

from candidate import topo_sort


def assert_valid_order(graph, order):
    assert len(order) == len(set(order)), "nodes must not repeat"
    nodes = set(graph) | {t for targets in graph.values() for t in targets}
    assert set(order) == nodes, "every node must appear exactly once"
    position = {node: i for i, node in enumerate(order)}
    for node, targets in graph.items():
        for target in targets:
            assert position[node] < position[target], f"{node} must precede {target}"


def test_example_from_the_prompt():
    graph = {"a": ["b"], "b": ["c"], "c": []}
    assert topo_sort(graph) == ["a", "b", "c"]


def test_empty_graph():
    assert topo_sort({}) == []


def test_single_node():
    assert topo_sort({"a": []}) == ["a"]


def test_diamond_dependency():
    graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    assert_valid_order(graph, topo_sort(graph))


def test_nodes_appearing_only_as_targets_are_included():
    """'c' is never a key, but it is still a node in the graph."""
    graph = {"a": ["b"], "b": ["c"]}
    order = topo_sort(graph)
    assert "c" in order
    assert_valid_order(graph, order)


def test_disconnected_components():
    graph = {"a": ["b"], "b": [], "x": ["y"], "y": []}
    assert_valid_order(graph, topo_sort(graph))


def test_cycle_raises_value_error():
    with pytest.raises(ValueError):
        topo_sort({"a": ["b"], "b": ["a"]})


def test_cycle_message_names_a_node_in_the_cycle():
    with pytest.raises(ValueError) as info:
        topo_sort({"a": ["b"], "b": ["c"], "c": ["a"]})
    assert any(node in str(info.value) for node in ("a", "b", "c"))


def test_self_loop_is_a_cycle():
    with pytest.raises(ValueError):
        topo_sort({"a": ["a"]})


def test_large_chain_is_linear():
    graph = {str(i): [str(i + 1)] for i in range(5000)}
    graph["5000"] = []
    assert_valid_order(graph, topo_sort(graph))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
