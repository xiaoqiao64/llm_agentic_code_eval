from serializers import to_dict


class User:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def as_dict(self):
        return to_dict(self)
