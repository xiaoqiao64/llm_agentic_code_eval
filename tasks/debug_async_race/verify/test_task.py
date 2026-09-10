import asyncio

from counter import AsyncCounter


def test_increment_concurrent():
    async def run():
        c = AsyncCounter()
        await asyncio.gather(*[c.increment() for _ in range(100)])
        return c.value()

    assert asyncio.run(run()) == 100
