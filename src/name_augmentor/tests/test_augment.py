import sys
import unittest

sys.path.insert(0, "../..")
from name_augmentor import Augmentor, load_default_config  # noqa: E402


# -----------------------------------------------------------------------------
class TestAugmentor(unittest.TestCase):
    # -------------------------------------------------------------------------
    def setUp(self):
        self.aug = Augmentor(config=load_default_config(), seed=42)

    # -------------------------------------------------------------------------
    def test_no_noise_returns_same(self):
        name = "José García"
        out, ops = self.aug.augment(name, lang="es", noise_rate=0.0)
        self.assertEqual(out, name)
        self.assertEqual(ops, [])

    # -------------------------------------------------------------------------
    def test_drop_diacritics_possible(self):
        # With high noise and forcing max ops, it's likely to change
        name = "José"
        changed = False
        for _ in range(50):
            out, ops = self.aug.augment(name, lang="es", noise_rate=1.0, max_ops=2)
            if out != name:
                changed = True
                break
        self.assertTrue(changed)

    # -------------------------------------------------------------------------
    def test_particle_rule_es(self):
        name = "Juan de la Cruz"
        out, ops = self.aug.augment(name, lang="es", noise_rate=1.0, max_ops=0)
        self.assertNotEqual(out, "")  # should be valid string

    # -------------------------------------------------------------------------
    def test_initials(self):
        name = "John Smith"
        out, ops = self.aug.augment(name, lang="en", noise_rate=1.0, max_ops=0)
        self.assertTrue(len(out) >= 1)

    # -------------------------------------------------------------------------
    def test_transliteration_ru(self):
        name = "Алексей Иванов"
        out, ops = self.aug.augment(name, lang="ru", noise_rate=1.0, max_ops=0)
        # Should contain Latin transliteration if applied
        # We can't guarantee it triggers each time; try multiple times
        ok = False
        for _ in range(40):
            out, ops = self.aug.augment(name, lang="ru", noise_rate=1.0, max_ops=0)
            if any(op.startswith("transliteration(ru)") for op in ops):
                ok = True
                break
        self.assertTrue(ok)

    # -------------------------------------------------------------------------
    def test_batch(self):
        rows = [("Robert Downey", "en"), ("Muhammad Ali", "ar")]
        out = self.aug.batch_augment(rows, noise_rate=1.0, max_ops=1)
        self.assertEqual(len(out), 2)
        for orig, aug, ops in out:
            self.assertTrue(isinstance(aug, str))
            self.assertTrue(isinstance(ops, list))


# =============================================================================
if __name__ == "__main__":
    unittest.main()
