class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value) -> None:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError
