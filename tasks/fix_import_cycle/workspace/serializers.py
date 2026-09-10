from models import User


def to_dict(user: User) -> dict:
    return {"name": user.name, "age": user.age}
