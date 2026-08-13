"""Reference tests for the LRUCache task."""

import time

import pytest

from candidate import LRUCache


def test_get_and_put_round_trip():
    cache = LRUCache(2)
    cache.put("a", 1)
    assert cache.get("a") == 1


def test_missing_key_returns_minus_one():
    assert LRUCache(2).get("nope") == -1


def test_evicts_when_over_capacity():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") == -1
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_get_counts_as_a_use():
    """The distinction between least-recently-used and oldest-inserted."""
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' is now the most recently used
    cache.put("c", 3)  # so 'b' should be evicted, not 'a'
    assert cache.get("a") == 1
    assert cache.get("b") == -1


def test_update_counts_as_a_use_and_does_not_grow_the_cache():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)  # refreshes 'a'
    cache.put("c", 3)  # evicts 'b'
    assert cache.get("a") == 10
    assert cache.get("b") == -1
    assert cache.get("c") == 3


def test_capacity_of_one():
    cache = LRUCache(1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == -1
    assert cache.get("b") == 2


def test_repeated_updates_never_exceed_capacity():
    cache = LRUCache(3)
    for i in range(100):
        cache.put(i % 3, i)
    assert all(cache.get(k) != -1 for k in range(3))


def test_operations_are_constant_time():
    """A linear scan per operation makes this take quadratic time."""
    cache = LRUCache(1000)
    start = time.monotonic()
    for i in range(50000):
        cache.put(i, i)
        cache.get(i)
    assert time.monotonic() - start < 5.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
