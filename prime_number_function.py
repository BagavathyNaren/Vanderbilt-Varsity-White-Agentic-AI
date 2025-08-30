
def is_prime(n):
    """
    Determine whether a given integer is a prime number.

    A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.

    Parameters
    ----------
    n : int
        The integer to be checked for primality.

    Returns
    -------
    bool
        True if `n` is a prime number, False otherwise.

    Examples
    --------
    >>> is_prime(2)
    True
    >>> is_prime(17)
    True
    >>> is_prime(18)
    False
    >>> is_prime(-5)
    False
    >>> is_prime(1)
    False
        
    Edge Cases
    ----------
    - Values less than or equal to 1 are not prime.
    - 2 and 3 are the smallest prime numbers.
    - Even numbers greater than 2 are not prime.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


import unittest

def is_prime(n):
    """
    Determine whether a given integer is a prime number.

    A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.

    Parameters
    ----------
    n : int
        The integer to be checked for primality.

    Returns
    -------
    bool
        True if `n` is a prime number, False otherwise.

    Examples
    --------
    >>> is_prime(2)
    True
    >>> is_prime(17)
    True
    >>> is_prime(18)
    False
    >>> is_prime(-5)
    False
    >>> is_prime(1)
    False
        
    Edge Cases
    ----------
    - Values less than or equal to 1 are not prime.
    - 2 and 3 are the smallest prime numbers.
    - Even numbers greater than 2 are not prime.
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class TestIsPrime(unittest.TestCase):
    def test_basic_primes(self):
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertTrue(is_prime(17))
        self.assertTrue(is_prime(97))

    def test_basic_non_primes(self):
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(-7))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(100))

    def test_large_prime(self):
        self.assertTrue(is_prime(7919))  # 7919 is prime

    def test_large_non_prime(self):
        self.assertFalse(is_prime(8000))  # 8000 is not prime

    def test_even_numbers(self):
        for even in [4, 6, 8, 10, 1000, 2002]:
            self.assertFalse(is_prime(even))

    def test_type_error(self):
        with self.assertRaises(TypeError):
            is_prime(3.5)
        with self.assertRaises(TypeError):
            is_prime("13")
        with self.assertRaises(TypeError):
            is_prime(None)
        with self.assertRaises(TypeError):
            is_prime([7])

    def test_edge_cases(self):
        self.assertFalse(is_prime(-1))
        self.assertFalse(is_prime(-2))
        self.assertFalse(is_prime(-3))
        self.assertFalse(is_prime(1))
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))

if __name__ == "__main__":
    unittest.main()