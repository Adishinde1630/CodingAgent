from app import get_user, register_user
from models import UserStore


def make_payload(**overrides):
    payload = {"username": "alice", "email": "alice@example.com", "password": "secret123"}
    payload.update(overrides)
    return payload


def test_register_success():
    store = UserStore()
    status, body = register_user(make_payload(), store)
    assert status == 201
    assert body["user"]["username"] == "alice"
    assert "password_hash" not in body["user"]
    assert store.count() == 1


def test_register_missing_fields():
    status, body = register_user({"username": "alice"}, UserStore())
    assert status == 400
    assert "required" in body["error"]


def test_register_weak_password():
    status, _ = register_user(make_payload(password="short"), UserStore())
    assert status == 400


def test_register_duplicate_username():
    store = UserStore()
    register_user(make_payload(), store)
    status, _ = register_user(make_payload(email="other@example.com"), store)
    assert status == 409


def test_get_user():
    store = UserStore()
    _, body = register_user(make_payload(), store)
    status, found = get_user(body["user"]["id"], store)
    assert status == 200
    assert found["user"]["email"] == "alice@example.com"
    assert get_user(999, store)[0] == 404
