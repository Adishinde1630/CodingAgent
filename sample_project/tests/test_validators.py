from validators import is_strong_password, is_valid_username


def test_valid_usernames():
    assert is_valid_username("alice")
    assert is_valid_username("bob_42")


def test_invalid_usernames():
    assert not is_valid_username("ab")
    assert not is_valid_username("has space")
    assert not is_valid_username("")


def test_password_strength():
    assert is_strong_password("secret123")
    assert not is_strong_password("short1")
    assert not is_strong_password("onlyletters")
    assert not is_strong_password("12345678")
