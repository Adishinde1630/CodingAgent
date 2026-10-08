"""Data models and a tiny in-memory store for the sample user service."""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class User:
    id: int
    username: str
    email: str
    password_hash: str
    created_at: str

    def to_public_dict(self) -> dict:
        """Return the user as a dict that is safe to send to API clients."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
        }


class UserStore:
    """In-memory user storage (stand-in for a real database)."""

    def __init__(self) -> None:
        self._users: dict[int, User] = {}
        self._next_id = 1

    def add(self, username: str, email: str, password_hash: str) -> User:
        user = User(
            id=self._next_id,
            username=username,
            email=email,
            password_hash=password_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._users[user.id] = user
        self._next_id += 1
        return user

    def get(self, user_id: int) -> Optional[User]:
        return self._users.get(user_id)

    def get_by_username(self, username: str) -> Optional[User]:
        for user in self._users.values():
            if user.username.lower() == username.lower():
                return user
        return None

    def count(self) -> int:
        return len(self._users)
