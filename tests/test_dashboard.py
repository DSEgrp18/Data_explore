import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from app import clean_input, integer_to_sinhala, split_text


class DashboardTextTests(unittest.TestCase):
    def test_markdown_is_removed_but_label_remains(self):
        value = clean_input("[ශ්‍රී ලංකාව](https://example.test) *සිංහල*", False)
        self.assertEqual(value, "ශ්‍රී ලංකාව සිංහල")

    def test_numbers_are_expanded(self):
        self.assertEqual(clean_input("මිලියන 20 සහ 3", True), "මිලියන විස්ස සහ තුන")

    def test_integer_ranges(self):
        self.assertEqual(integer_to_sinhala(250), "දෙසිය පනහ")
        self.assertEqual(integer_to_sinhala(2026), "දෙදහස් විසිහය")

    def test_long_text_is_chunked(self):
        chunks = split_text("වචනය " * 80, max_chars=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 40 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
