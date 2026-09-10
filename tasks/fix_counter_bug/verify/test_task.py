from counter import Counter


def test_increment():
    c = Counter()
    assert c.increment() == 1
    assert c.value() == 1
    assert c.increment() == 2
    assert c.value() == 2


def test_decrement():
    c = Counter()
    c.increment()
    c.increment()
    assert c.decrement() == 1
    assert c.value() == 1


def test_reset():
    c = Counter()
    c.increment()
    c.increment()
    c.reset()
    assert c.value() == 0
