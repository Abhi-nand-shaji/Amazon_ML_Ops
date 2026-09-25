"""Regression tests for behaviors that were found (and fixed) during development.

Run:  python3 -m unittest discover -s code/business_entity_resolution/tests -v
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text_normalize import normalize_address, normalize_name  # noqa: E402


def name(s):
    return {k: v.iloc[0] for k, v in normalize_name(pd.Series([s])).items()}


def addr(s):
    return {k: v.iloc[0] for k, v in normalize_address(pd.Series([s])).items()}


class NameNormalizationTest(unittest.TestCase):
    def test_accent_noise_is_folded(self):
        # the largest non-ASCII group in the data is Latin text with injected accents
        self.assertEqual(name("Payne Énterprises")["name_normalized"], "payne enterprises")
        self.assertEqual(name("Lumay Bóral")["name_normalized"], "lumay boral")

    def test_ampersand_punctuation_and_suffix_variants_collapse(self):
        a = name("Chordia &-Pártners Ltd")["name_normalized"]
        b = name("Chordia and Partners Limited")["name_normalized"]
        self.assertEqual(a, b)
        self.assertEqual(name("Ram Marketing Private Limited")["name_normalized"], "ram marketing pvt ltd")

    def test_core_drops_legal_suffix_but_keeps_information_elsewhere(self):
        r = name("Acme Robotics Inc")
        self.assertEqual(r["name_core"], "acme robotics")
        self.assertEqual(r["name_normalized"], "acme robotics inc")  # suffix preserved in the full form

    def test_token_order_representation(self):
        self.assertEqual(name("Robotics Acme")["name_sorted_tokens"], name("Acme Robotics")["name_sorted_tokens"])

    def test_devanagari_is_transliterated_to_latin(self):
        out = name("राम मार्केटिंग प्राइवेट लिमिटेड")
        self.assertEqual(out["name_script"], "devanagari")
        self.assertTrue(out["name_normalized"].isascii())
        self.assertIn("limiteda", out["name_normalized"])  # 'Limited' -> near-identical Latin token

    def test_other_indic_scripts_are_preserved_not_wiped(self):
        # regression: an ASCII-only punctuation regex used to turn Tamil names into ''
        out = name("குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்")
        self.assertEqual(out["name_script"], "tamil")
        self.assertGreater(len(out["name_normalized"].replace(" ", "")), 0)

    def test_missing_values_do_not_crash(self):
        self.assertEqual(name(None)["name_normalized"], "")
        self.assertEqual(name("")["name_core"], "")


class AddressNormalizationTest(unittest.TestCase):
    def test_street_type_abbreviations(self):
        self.assertEqual(addr("1795 Westchester Drive, High Point, NC")["address_normalized"], "1795 westchester dr high point nc")
        self.assertEqual(addr("22 Pine Road")["address_normalized"], addr("22 Pine Rd")["address_normalized"])

    def test_postal_code_and_house_number_extraction(self):
        r = addr("175 Boulevard du Président Franklin Roosevelt, Bordeaux, 33000")
        self.assertEqual(r["postal_code"], "33000")
        self.assertEqual(r["house_number"], "175")
        self.assertTrue(pd.isna(addr("No number here")["house_number"]))
        self.assertTrue(pd.isna(addr("Main St, Austin")["postal_code"]))

    def test_accents_folded_in_addresses(self):
        self.assertIn("president", addr("175 Boulevard du Président Franklin Roosevelt")["address_normalized"])

    def test_missing_address(self):
        r = addr(None)
        self.assertEqual(r["address_normalized"], "")
        self.assertTrue(pd.isna(r["postal_code"]))


if __name__ == "__main__":
    unittest.main()
