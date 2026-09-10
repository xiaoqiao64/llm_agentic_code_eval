from lru_cache import LRUCache


def test_basic():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)
    assert c.get("b") is None
    assert c.get("c") == 3


def test_update_moves_to_recent():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")
    c.put("c", 3)
    assert c.get("b") is None
    assert c.get("a") == 1


def test_size():
    c = LRUCache(3)
    c.put("x", 10)
    c.put("y", 20)
    assert c.size() == 2
