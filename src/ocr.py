"""OCR nomor ijazah dengan Tesseract + metrik CER."""
import re
import cv2
import numpy as np
import pytesseract

from .config import NOMOR_REGEX


def _pad(img, p=20):
    v = int(np.median(img))
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=v)


def read_number(roi_img: np.ndarray):
    """Jalankan OCR pada ROI (sudah di-enhance). Return (nomor, teks_mentah)."""
    img = _pad(roi_img)
    raw = ""
    for psm in (7, 6):
        raw = pytesseract.image_to_string(
            img, lang="eng", config=f"--oem 3 --psm {psm}").strip()
        # koreksi karakter yang sering tertukar pada deret angka
        fixed = raw.translate(str.maketrans("OoIlSB", "001158"))
        m = re.findall(NOMOR_REGEX, re.sub(r"[\s.,\-]", "", fixed))
        if m:
            return max(m, key=len), raw
    return "", raw


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    """Character Error Rate = (S + D + I) / N, N = panjang ground truth."""
    return levenshtein(pred, truth) / max(len(truth), 1)
