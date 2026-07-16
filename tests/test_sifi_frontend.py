import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sifi_frontend import extract_features, grapheme_clusters


class FrontendTests(unittest.TestCase):
    def test_vowel_sign_stays_with_consonant(self):
        self.assertEqual(grapheme_clusters("කා"), ["කා"])
        item = extract_features("කා")[0]
        self.assertEqual((item.base, item.vowel, item.vowel_length), ("ක", "a", "long"))

    def test_inherent_vowel(self):
        item = extract_features("ක")[0]
        self.assertTrue(item.inherent_vowel)
        self.assertEqual(item.vowel, "a")

    def test_virama_removes_inherent_vowel(self):
        item = extract_features("ක්")[0]
        self.assertFalse(item.inherent_vowel)
        self.assertEqual(item.vowel, "")

    def test_aspiration_and_prenasalization(self):
        aspirated = extract_features("ඛ")[0]
        prenasalized = extract_features("ඟ")[0]
        self.assertTrue(aspirated.aspirated)
        self.assertTrue(prenasalized.prenasalized)

    def test_zwj_conjunct_is_one_cluster(self):
        clusters = grapheme_clusters("ශ්‍රී")
        self.assertEqual(clusters, ["ශ්‍රී"])
        self.assertTrue(extract_features("ශ්‍රී")[0].conjunct)

    def test_code_switch_flag(self):
        latin = [item for item in extract_features("AI") if item.code_switch]
        self.assertEqual(len(latin), 2)


if __name__ == "__main__":
    unittest.main()
