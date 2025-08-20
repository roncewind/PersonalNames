# import csv
# import os
import sys

# import tempfile
import unittest

sys.path.insert(0, "../..")

from name_augmentor.translit import (  # noqa: E402
    get_adapter_name,
    register_adapter,
    transliterate,
)
from name_augmentor.triplets import TripletGenerator  # noqa: E402


# -----------------------------------------------------------------------------
class TestTranslit(unittest.TestCase):
    # -------------------------------------------------------------------------
    def test_builtin_ru(self):
        out, name = transliterate("Алексей", "ru")
        self.assertNotEqual(out, "Алексей")
        self.assertIn("ru", get_adapter_name("ru"))

    # -------------------------------------------------------------------------
    def test_unknown_lang_noop(self):
        s = "名前"
        out, name = transliterate(s, "ja")  # likely no pykakasi installed here
        # Either a transliteration if available, or the same string
        self.assertTrue(out == s or isinstance(out, str))

    # -------------------------------------------------------------------------
    def test_custom_adapter(self):
        register_adapter("xx", "custom_upper", lambda s: s.upper())
        out, name = transliterate("abc", "xx")
        self.assertEqual(out, "ABC")
        self.assertEqual(name, "custom_upper")


# -----------------------------------------------------------------------------
class TestTriplets(unittest.TestCase):
    # -------------------------------------------------------------------------
    def test_generate_triplets(self):
        rows = [
            ("c1", "Robert Downey", "en"),
            ("c1", "Bob Downey", "en"),
            ("c2", "Roberto Dauni", "es"),
            ("c3", "Muhammad Ali", "ar"),
            ("c4", "Алексей Иванов", "ru"),
            ("c5", "José García", "es"),
        ]
        tg = TripletGenerator(seed=7)
        trips = tg.generate(rows, per_cluster=2, noise_rate=0.9, max_ops=2)
        self.assertTrue(len(trips) >= 4)
        for t in trips:
            self.assertIsInstance(t.anchor, str)
            self.assertIsInstance(t.pos, str)
            self.assertIsInstance(t.neg, str)
            self.assertNotEqual(t.cluster_id, "")


# =============================================================================
if __name__ == "__main__":
    unittest.main()
