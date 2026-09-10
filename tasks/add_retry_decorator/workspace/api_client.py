_failures_remaining = 2


def fetch_data() -> str:
    global _failures_remaining
    if _failures_remaining > 0:
        _failures_remaining -= 1
        raise ConnectionError("transient failure")
    return "ok"
