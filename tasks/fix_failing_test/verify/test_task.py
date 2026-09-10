from fib import fibonacci


def test_base_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


def test_sequence():
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55


def test_larger():
    assert fibonacci(20) == 6765
