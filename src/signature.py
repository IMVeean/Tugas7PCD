"""Deteksi tanda tangan: Thresholding -> Morphology -> Signature Detection."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class SigResult:
    present: bool
    ink_ratio: float      # proporsi piksel tinta pada ROI
    width_ratio: float    # lebar bbox komponen terbesar / lebar ROI
    height_ratio: float   # tinggi bbox komponen terbesar / tinggi ROI
    n_components: int
    mask: np.ndarray


def threshold_ink(gray_roi: np.ndarray) -> np.ndarray:
    """Thresholding adaptif-terhadap-background.

    Ambang = min(Otsu, median_background - 55). Syarat kedua mencegah Otsu
    "memaksa" memisahkan tekstur kertas ketika ROI sebenarnya kosong
    (tanpa tinta), sehingga ROI kosong -> hampir tidak ada piksel foreground.
    """
    g = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg = float(np.median(g))
    thr = min(otsu, bg - 55)
    return (g < thr).astype(np.uint8) * 255


def clean_morphology(mask: np.ndarray) -> np.ndarray:
    """Morphology: opening (buang noise titik) -> closing (sambung goresan)."""
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)


def detect_signature(gray_roi: np.ndarray,
                     min_ink=0.012, min_w=0.30, min_h=0.18,
                     min_area_ratio=0.004) -> SigResult:
    """Tentukan ada/tidaknya tanda tangan pada ROI.

    Tanda tangan = goresan tinta kontinu yang relatif BESAR (lebar & tinggi
    signifikan terhadap ROI), berbeda dengan teks cetak yang berupa komponen
    kecil-kecil. PRESENT jika komponen terhubung terbesar punya bbox
    >= min_w x min_h ROI dan rasio tinta total >= min_ink.
    """
    mask = clean_morphology(threshold_ink(gray_roi))
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = min_area_ratio * h * w
    comps = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= min_area]
    ink_ratio = float((mask > 0).sum()) / (h * w)
    if not comps:
        return SigResult(False, ink_ratio, 0.0, 0.0, 0, mask)
    big = max(comps, key=lambda s: s[cv2.CC_STAT_WIDTH] * s[cv2.CC_STAT_HEIGHT])
    wr = big[cv2.CC_STAT_WIDTH] / w
    hr = big[cv2.CC_STAT_HEIGHT] / h
    present = bool(wr >= min_w and hr >= min_h and ink_ratio >= min_ink)
    return SigResult(present, ink_ratio, float(wr), float(hr), len(comps), mask)
