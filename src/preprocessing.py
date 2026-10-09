"""Tahap awal pipeline: orientasi, grayscale, enhancement global, crop ROI."""
import cv2
import numpy as np
import pytesseract


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Konversi BGR -> Grayscale."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _orientation_score(gray: np.ndarray) -> float:
    """Skor keterbacaan teks: jumlah confidence kata alfabet >=3 huruf."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    d = pytesseract.image_to_data(small, lang="eng", config="--psm 6",
                                  output_type=pytesseract.Output.DICT)
    score = 0.0
    for t, c in zip(d["text"], d["conf"]):
        t = t.strip()
        try:
            c = float(c)
        except ValueError:
            continue
        if len(t) >= 3 and t.isalpha() and c > 50:
            score += c
    return score


def auto_orient(img_bgr: np.ndarray):
    """Pilih rotasi (0/90/180/270) yang paling terbaca oleh OCR.

    Returns: (citra_terotasi, derajat_rotasi_searah_jarum_jam)
    """
    rots = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    best, best_s = 0, -1
    for deg, code in rots.items():
        im = img_bgr if code is None else cv2.rotate(img_bgr, code)
        s = _orientation_score(to_gray(im))
        if s > best_s:
            best, best_s = deg, s
    code = rots[best]
    return (img_bgr if code is None else cv2.rotate(img_bgr, code)), best


def global_enhance(gray: np.ndarray) -> np.ndarray:
    """Image enhancement global:
    1) Normalisasi iluminasi: bagi citra dengan estimasi background
       (morphological closing + blur) -> meratakan warna kuning/tekstur kertas.
    2) CLAHE ringan untuk menaikkan kontras lokal.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    bg = cv2.GaussianBlur(bg, (0, 0), 21)
    norm = cv2.divide(gray, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm)


def crop_rel(img: np.ndarray, roi) -> np.ndarray:
    """Crop berdasarkan ROI relatif (x1,y1,x2,y2)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)].copy()
