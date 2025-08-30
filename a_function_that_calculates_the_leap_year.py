
def is_leap_year(year):
    """
    Determine if a given year is a leap year.

    A year is a leap year if it is divisible by 4 but not by 100,
    except when it is divisible by 400.

    Parameters:
        year (int): The year to check. Should be a positive integer.

    Returns:
        bool: True if the year is a leap year, False otherwise.

    Examples:
        >>> is_leap_year(2020)
        True
        >>> is_leap_year(1900)
        False
        >>> is_leap_year(2000)
        True
        >>> is_leap_year(2023)
        False

    Edge Cases:
        - Year 0 is considered a leap year by the algorithm as it is divisible by 400.
        - Negative years might not be meaningful in the Gregorian calendar but the function will still apply the logic.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


import unittest

def is_leap_year(year):
    """
    Determine if a given year is a leap year.

    A year is a leap year if it is divisible by 4 but not by 100,
    except when it is divisible by 400.

    Parameters:
        year (int): The year to check. Should be a positive integer.

    Returns:
        bool: True if the year is a leap year, False otherwise.

    Examples:
        >>> is_leap_year(2020)
        True
        >>> is_leap_year(1900)
        False
        >>> is_leap_year(2000)
        True
        >>> is_leap_year(2023)
        False

    Edge Cases:
        - Year 0 is considered a leap year by the algorithm as it is divisible by 400.
        - Negative years might not be meaningful in the Gregorian calendar but the function will still apply the logic.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


class TestIsLeapYear(unittest.TestCase):
    def test_basic_functionality(self):
        self.assertTrue(is_leap_year(2020))  # divisible by 4, not by 100
        self.assertFalse(is_leap_year(2019)) # not divisible by 4
        self.assertFalse(is_leap_year(1900)) # divisible by 100 but not 400
        self.assertTrue(is_leap_year(2000))  # divisible by 400

    def test_edge_cases(self):
        self.assertTrue(is_leap_year(0))     # year zero edge case, divisible by 400
        self.assertTrue(is_leap_year(-400))  # negative year divisible by 400
        self.assertFalse(is_leap_year(-100)) # negative year divisible by 100 but not 400
        self.assertFalse(is_leap_year(-3))   # negative not divisible by 4

    def test_large_years(self):
        self.assertTrue(is_leap_year(400000))
        self.assertFalse(is_leap_year(400001))

    def test_non_integer_input(self):
        # As the function does not do type checking, passing non-integers may cause error or incorrect results.
        with self.assertRaises(TypeError):
            is_leap_year("2000")
        with self.assertRaises(TypeError):
            is_leap_year(2000.0)  # float instead of int

    def test_zero_division_error(self):
        # Should not raise zero division error or similar because of modulo use
        self.assertFalse(is_leap_year(3))
        

if __name__ == "__main__":
    unittest.main()