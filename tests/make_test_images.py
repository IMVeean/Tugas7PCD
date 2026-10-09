"""Membuat citra uji TANPA tanda tangan (kontrol negatif) dari sample ijazah.

Tinta tanda tangan dihapus dengan cv2.inpaint, sehingga kertas tampak bersih.
  data/ijazah_tanpa_ttd.jpg   : TTD Rektor & Dekan dihapus
  data/ijazah_hanya_dekan.jpg : hanya TTD Rektor dihapus
"""
import os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.preprocessing import auto_orient, to_gray
from src.signature import threshold_ink

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def erase(img, roi):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    a, b, c, d = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    g = to_gray(img)[b:d, a:c]
    ink = threshold_ink(g)
    ink = cv2.dilate(ink, np.ones((9, 9), np.uint8))
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[b:d, a:c] = ink
    # biarkan teks cetak ("Rektor","Dekan",nama) tetap ada: hanya ROI yang diproses
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


img, _ = auto_orient(cv2.imread(os.path.join(ROOT, "data", "ijazah_001.jpg")))
cv2.imwrite(os.path.join(ROOT, "data", "ijazah_hanya_dekan.jpg"),
            erase(img, C.ROI_TTD_REKTOR))
no = erase(erase(img, C.ROI_TTD_REKTOR), C.ROI_TTD_DEKAN)
cv2.imwrite(os.path.join(ROOT, "data", "ijazah_tanpa_ttd.jpg"), no)
print("OK")
