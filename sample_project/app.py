"""User registration API (framework-free so it is easy to read and test).

Each handler returns a ``(status_code, json_payload)`` tuple, like a tiny
HTTP endpoint would.
"""
import hashlib
from typing import Optional

from models import UserStore
from validators import is_strong_password, is_valid_username

default_store = UserStore()


def hash_password(password: str) -> str:
    # Demo only - use bcrypt/argon2 in a real application.
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(data: dict, store: Optional[UserStore] = None) -> tuple[int, dict]:
    """POST /register - user registration: create a new account from a JSON body."""
    if store is None:
        store = default_store

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return 400, {"error": "username, email and password are required"}

    if not is_valid_username(username):
        return 400, {"error": "Invalid username"}

    if not is_strong_password(password):
        return 400, {"error": "Password must be 8+ characters with letters and digits"}

    if store.get_by_username(username):
        return 409, {"error": "Username already taken"}

    user = store.add(username, email, hash_password(password))
    return 201, {"user": user.to_public_dict()}


def get_user(user_id: int, store: Optional[UserStore] = None) -> tuple[int, dict]:
    """GET /users/<id> - fetch a user's public profile."""
    if store is None:
        store = default_store
    user = store.get(user_id)
    if user is None:
        return 404, {"error": "User not found"}
    return 200, {"user": user.to_public_dict()}
