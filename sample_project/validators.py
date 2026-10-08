"""Input validation helpers for the user service."""
import re

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,20}$")


def is_valid_username(username: str) -> bool:
    """A username is 3-20 characters: letters, digits or underscores."""
    return isinstance(username, str) and bool(USERNAME_PATTERN.match(username))


def is_strong_password(password: str) -> bool:
    """A strong password has 8+ characters, at least one letter and one digit."""
    if not isinstance(password, str) or len(password) < 8:
        return False
    has_letter = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    return has_letter and has_digit
