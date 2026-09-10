import asyncio


class AsyncCounter:
    def __init__(self) -> None:
        self._value = 0

    async def increment(self) -> None:
        await asyncio.sleep(0)
        current = self._value
        await asyncio.sleep(0)
        self._value = current + 1

    def value(self) -> int:
        return self._value
