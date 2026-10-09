"""Uji sederhana: python -m unittest discover tests -v"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import verify
from src.ocr import cer

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class T(unittest.TestCase):
    def test_cer(self):
        self.assertEqual(cer("123", "123"), 0.0)
        self.assertAlmostEqual(cer("124", "123"), 1 / 3)

    def test_sample(self):
        r = verify(os.path.join(D, "ijazah_001.jpg"))
        self.assertEqual(r["nomor_ijazah"], "571012022000056")
        self.assertEqual(r["tanda_tangan"], "PRESENT")

    def test_tanpa_ttd(self):
        r = verify(os.path.join(D, "ijazah_tanpa_ttd.jpg"))
        self.assertEqual(r["tanda_tangan"], "ABSENT")


if __name__ == "__main__":
    unittest.main()
