"""Metode enhancement untuk ROI nomor ijazah (dibandingkan lewat CER)."""
import cv2


def m_baseline(g):                      # hanya grayscale (pembanding)
    return g


def m_hist_eq(g):                       # Histogram Equalization global
    return cv2.equalizeHist(g)


def m_clahe(g):                         # CLAHE
    return cv2.createCLAHE(2.0, (4, 4)).apply(g)


def m_unsharp(g):                       # Gaussian smoothing + Unsharp mask
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.addWeighted(g, 1.8, cv2.GaussianBlur(blur, (0, 0), 2.0), -0.8, 0)


def m_otsu(g):                          # Otsu threshold
    _, t = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def m_adaptive(g):                      # Adaptive Gaussian threshold
    return cv2.adaptiveThreshold(cv2.GaussianBlur(g, (3, 3), 0), 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)


def m_clahe_bilateral_otsu(g):          # CLAHE + Bilateral + Otsu
    c = cv2.createCLAHE(2.0, (4, 4)).apply(g)
    b = cv2.bilateralFilter(c, 7, 50, 50)
    _, t = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def m_upscale_clahe_sharp(g):           # Upscale 2x (cubic) + CLAHE + sharpen
    up = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    c = cv2.createCLAHE(2.0, (4, 4)).apply(up)
    blur = cv2.GaussianBlur(c, (0, 0), 1.2)
    return cv2.addWeighted(c, 1.6, blur, -0.6, 0)


def m_upscale_denoise_otsu(g):          # Upscale + median denoise + Otsu
    up = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    up = cv2.medianBlur(up, 3)
    _, t = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


METHODS = {
    "baseline_gray": m_baseline,
    "hist_equalization": m_hist_eq,
    "clahe": m_clahe,
    "unsharp": m_unsharp,
    "otsu": m_otsu,
    "adaptive_thresh": m_adaptive,
    "clahe_bilateral_otsu": m_clahe_bilateral_otsu,
    "upscale_clahe_sharpen": m_upscale_clahe_sharp,
    "upscale_median_otsu": m_upscale_denoise_otsu,
}

DEFAULT_METHOD = "adaptive_thresh"  # CER rata-rata terendah (lihat evaluate.py)
