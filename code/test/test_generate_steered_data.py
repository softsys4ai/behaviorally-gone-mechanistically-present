import unittest
import sys
import os

# Add src to the path to make importing easier
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import MagicMock, patch

for mod in ["torch", "transformers", "peft", "numpy", "tqdm", "datasets"]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from generate_steered_data import (
    extract_three_digit_numbers_consistent_sep,
    validate_completion
)

class TestExtractThreeDigitNumbersConsistentSep(unittest.TestCase):

    def test_empty_or_no_numbers(self):
        self.assertIsNone(extract_three_digit_numbers_consistent_sep(""))
        self.assertIsNone(extract_three_digit_numbers_consistent_sep("hello world"))
        self.assertIsNone(extract_three_digit_numbers_consistent_sep("12 1234 56789"))

    def test_single_three_digit_number(self):
        self.assertEqual(extract_three_digit_numbers_consistent_sep("123"), [123])
        self.assertEqual(extract_three_digit_numbers_consistent_sep("Here is a number: 456."), [456])

    def test_consistent_separators(self):
        # Comma and space
        self.assertEqual(extract_three_digit_numbers_consistent_sep("123, 456, 789"), [123, 456, 789])
        # Just comma
        self.assertEqual(extract_three_digit_numbers_consistent_sep("123,456,789"), [123, 456, 789])
        # Hyphen
        self.assertEqual(extract_three_digit_numbers_consistent_sep("123-456-789"), [123, 456, 789])
        # Newlines
        self.assertEqual(extract_three_digit_numbers_consistent_sep("123\n456\n789"), [123, 456, 789])
        # Spaces
        self.assertEqual(extract_three_digit_numbers_consistent_sep("123 456 789"), [123, 456, 789])
        # Complex separator
        self.assertEqual(extract_three_digit_numbers_consistent_sep("123 - 456 - 789"), [123, 456, 789])

    def test_inconsistent_separators(self):
        # Comma+space then just comma
        self.assertIsNone(extract_three_digit_numbers_consistent_sep("123, 456,789"))
        # Comma then semicolon
        self.assertIsNone(extract_three_digit_numbers_consistent_sep("123, 456; 789"))
        # Newline then space
        self.assertIsNone(extract_three_digit_numbers_consistent_sep("123\n456 789"))
        # Mixed separators
        self.assertIsNone(extract_three_digit_numbers_consistent_sep("123 - 456 , 789"))

    def test_with_surrounding_text_consistent(self):
        # Text around the consistent numbers
        self.assertEqual(
            extract_three_digit_numbers_consistent_sep("Numbers are 111, 222, 333 today."),
            [111, 222, 333]
        )

    def test_interference_from_other_number_lengths(self):
        # Two 3-digit numbers separated by a specific string that happens to contain a 2-digit number
        # Matches: "123" and "456". Separator: " 12 "
        self.assertEqual(extract_three_digit_numbers_consistent_sep("123 12 456"), [123, 456])

        # If there's an inconsistent interference, it should fail
        # Matches: "123", "456", "789". Separators: " 12 " and " "
        self.assertIsNone(extract_three_digit_numbers_consistent_sep("123 12 456 789"))

class TestValidateCompletion(unittest.TestCase):

    @patch('generate_steered_data.extract_three_digit_numbers_consistent_sep')
    def test_no_numbers(self, mock_extract):
        mock_extract.return_value = None
        success, err_msg, cleaned = validate_completion("dummy", 1, 5)
        self.assertFalse(success)
        self.assertEqual(err_msg, "no 3-digit numbers with consistent separator")
        self.assertIsNone(cleaned)

    @patch('generate_steered_data.extract_three_digit_numbers_consistent_sep')
    def test_too_few_numbers(self, mock_extract):
        mock_extract.return_value = [123, 456]
        success, err_msg, cleaned = validate_completion("dummy", 3, 5)
        self.assertFalse(success)
        self.assertEqual(err_msg, "too few numbers (2 < 3)")
        self.assertIsNone(cleaned)

    @patch('generate_steered_data.extract_three_digit_numbers_consistent_sep')
    def test_too_many_numbers(self, mock_extract):
        mock_extract.return_value = [123, 456, 789, 101]
        success, err_msg, cleaned = validate_completion("dummy", 1, 3)
        self.assertFalse(success)
        self.assertEqual(err_msg, "too many numbers (4 > 3)")
        self.assertIsNone(cleaned)

    @patch('generate_steered_data.extract_three_digit_numbers_consistent_sep')
    def test_success(self, mock_extract):
        mock_extract.return_value = [123, 456, 789]
        success, err_msg, cleaned = validate_completion("dummy", 1, 5)
        self.assertTrue(success)
        self.assertIsNone(err_msg)
        self.assertEqual(cleaned, "123, 456, 789")

if __name__ == '__main__':
    unittest.main()
