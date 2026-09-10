def test_import_and_serialize():
    from models import User

    u = User("alice", 30)
    assert u.as_dict() == {"name": "alice", "age": 30}
