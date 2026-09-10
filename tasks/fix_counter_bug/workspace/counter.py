class Counter:
    def __init__(self) -> None:
        self._value = 0

    def increment(self) -> int:
        self._value += 1
        return self._value

    def decrement(self) -> int:
        self._value -= 1
        return self._value

    def reset(self) -> None:
        self._value = 0

    def value(self) -> int:
        # Bug: returns value before last operation semantics are wrong
        return self._value - 1
