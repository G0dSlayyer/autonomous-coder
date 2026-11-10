from solution import is_palindrome

def test_simple_palindrome():
    assert is_palindrome("racecar") == True

def test_not_palindrome():
    assert is_palindrome("hello") == False

def test_empty_string():
    assert is_palindrome("") == True

def test_case_insensitive():
    assert is_palindrome("RaceCar") == True
