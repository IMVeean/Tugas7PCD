"""Pipeline utama verifikasi ijazah."""
import os
import cv2
import numpy as np

from . import config as C
from .preprocessing import to_gray, auto_orient, global_enhance, crop_rel
from .enhancement import METHODS, DEFAULT_METHOD
from .ocr import read_number
from .signature import detect_signature


def _draw(img_bgr, roi, label, color):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = roi
    p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
    cv2.rectangle(img_bgr, p1, p2, color, 4)
    cv2.putText(img_bgr, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, color, 3, cv2.LINE_AA)


def verify(path, out_dir=None, method=DEFAULT_METHOD, require="any"):
    """Jalankan pipeline penuh pada satu citra ijazah.

    require: "any"  -> TTD = PRESENT bila minimal satu tanda tangan pejabat
                       (Rektor/Dekan) terdeteksi
             "all"  -> PRESENT hanya bila Rektor DAN Dekan terdeteksi
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Tidak bisa membaca citra: {path}")

    # 1) Orientasi -> 2) Grayscale -> 3) Enhancement global
    img, rot = auto_orient(img)
    gray = to_gray(img)
    enh = global_enhance(gray)

    # 4a) Jalur nomor: crop ROI -> enhancement -> OCR
    roi_no = crop_rel(gray, C.ROI_NOMOR)
    roi_no_enh = METHODS[method](roi_no)
    nomor, raw = read_number(roi_no_enh)

    # 4b) Jalur tanda tangan: crop ROI -> threshold -> morphology -> deteksi
    sigs = {
        "rektor": detect_signature(crop_rel(enh, C.ROI_TTD_REKTOR)),
        "dekan": detect_signature(crop_rel(enh, C.ROI_TTD_DEKAN)),
    }
    pemilik = detect_signature(crop_rel(enh, C.ROI_TTD_PEMILIK))
    flags = [s.present for s in sigs.values()]
    present = all(flags) if require == "all" else any(flags)

    result = {
        "file": os.path.basename(path),
        "rotasi_derajat": rot,
        "nomor_ijazah": nomor if nomor else "TIDAK TERBACA",
        "ocr_raw": raw,
        "tanda_tangan": "PRESENT" if present else "ABSENT",
        "detail_ttd": {k: dict(present=v.present, ink=round(v.ink_ratio, 4),
                              w=round(v.width_ratio, 3), h=round(v.height_ratio, 3))
                       for k, v in sigs.items()},
        "ttd_pemilik": pemilik.present,
        "metode_enhancement": method,
    }

    # Simpan artefak visual tiap tahap (untuk laporan)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        p = lambda n: os.path.join(out_dir, f"{base}_{n}.png")
        cv2.imwrite(p("1_gray"), gray)
        cv2.imwrite(p("2_enhanced"), enh)
        cv2.imwrite(p("3_roi_nomor"), roi_no)
        cv2.imwrite(p("4_roi_nomor_enh"), roi_no_enh)
        for k, s in sigs.items():
            r = C.ROI_TTD_REKTOR if k == "rektor" else C.ROI_TTD_DEKAN
            cv2.imwrite(p(f"5_ttd_{k}_roi"), crop_rel(enh, r))
            cv2.imwrite(p(f"6_ttd_{k}_mask"), s.mask)
        vis = img.copy()
        _draw(vis, C.ROI_NOMOR, "NOMOR", (255, 0, 0))
        _draw(vis, C.ROI_TTD_REKTOR, f"REKTOR:{'ADA' if sigs['rektor'].present else 'TIDAK'}",
              (0, 160, 0) if sigs['rektor'].present else (0, 0, 255))
        _draw(vis, C.ROI_TTD_DEKAN, f"DEKAN:{'ADA' if sigs['dekan'].present else 'TIDAK'}",
              (0, 160, 0) if sigs['dekan'].present else (0, 0, 255))
        cv2.imwrite(p("7_hasil_anotasi"),
                    cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    return result
