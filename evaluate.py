#!/usr/bin/env python3
"""Evaluasi metode enhancement berdasarkan CER (Character Error Rate).

Karena hanya ada satu citra bersih, ROI nomor ijazah diberi berbagai degradasi
sintetis (blur, noise, kontras rendah, iluminasi tidak merata, resolusi rendah,
kompresi JPEG) untuk meniru kondisi scan/foto di dunia nyata. Setiap metode
enhancement dijalankan pada tiap kondisi, lalu OCR dibandingkan dengan
ground truth -> CER.

Pemakaian:
    python evaluate.py                       # default: data/ijazah_001.jpg
    python evaluate.py --image data/x.jpg --truth 571012022000056
"""
import argparse, os, csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.preprocessing import auto_orient, to_gray, crop_rel
from src.enhancement import METHODS
from src.ocr import read_number, cer

rng = np.random.default_rng(42)  # reproducible


def d_clean(g):   return g
def d_blur(g):    return cv2.GaussianBlur(g, (0, 0), 2.2)
def d_noise(g):   return np.clip(g + rng.normal(0, 28, g.shape), 0, 255).astype(np.uint8)
def d_lowcon(g):  return np.clip(110 + 0.30 * (g.astype(np.float32) - 128) + 40, 0, 255).astype(np.uint8)
def d_uneven(g):
    h, w = g.shape
    ramp = np.tile(np.linspace(0.35, 1.15, w, dtype=np.float32), (h, 1))
    return np.clip(g * ramp, 0, 255).astype(np.uint8)
def d_lowres(g):
    s = cv2.resize(g, None, fx=0.30, fy=0.30, interpolation=cv2.INTER_AREA)
    return s
def d_jpeg(g):
    _, enc = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 6])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
def d_saltpepper(g):
    o = g.copy(); m = rng.random(g.shape)
    o[m < 0.03] = 0; o[m > 0.97] = 255
    return o
def d_combo(g):   return d_noise(d_blur(d_uneven(g)))

SCENARIOS = {"clean": d_clean, "blur": d_blur, "noise": d_noise,
             "low_contrast": d_lowcon, "uneven_light": d_uneven,
             "low_res": d_lowres, "jpeg_q6": d_jpeg,
             "salt_pepper": d_saltpepper, "kombinasi": d_combo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/ijazah_001.jpg")
    ap.add_argument("--truth", default=None, help="nomor ijazah ground truth")
    ap.add_argument("--out", default="output")
    ap.add_argument("--seeds", type=int, default=5, help="jumlah seed degradasi acak")
    a = ap.parse_args()
    truth = a.truth or C.GROUND_TRUTH[os.path.basename(a.image)]
    os.makedirs(a.out, exist_ok=True)

    img, _ = auto_orient(cv2.imread(a.image))
    roi = crop_rel(to_gray(img), C.ROI_NOMOR)

    # CER dirata-ratakan pada beberapa seed (noise acak) agar tidak bias satu sampel
    global rng
    table = {m: {s: 0.0 for s in SCENARIOS} for m in METHODS}
    for seed in range(a.seeds):
        rng = np.random.default_rng(42 + seed)
        for sname, sfun in SCENARIOS.items():
            deg = sfun(roi)
            for mname, mfun in METHODS.items():
                try:
                    pred, _ = read_number(mfun(deg))
                except Exception:
                    pred = ""
                table[mname][sname] += cer(pred, truth) / a.seeds

    # rata-rata CER per metode
    avg = {m: float(np.mean(list(v.values()))) for m, v in table.items()}
    ranking = sorted(avg, key=avg.get)

    # CSV
    scen = list(SCENARIOS)
    with open(os.path.join(a.out, "cer_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + scen + ["mean_CER"])
        for m in ranking:
            w.writerow([m] + [f"{table[m][s]:.3f}" for s in scen] + [f"{avg[m]:.3f}"])

    # Markdown
    lines = ["| Peringkat | Metode | " + " | ".join(scen) + " | **Rata-rata CER** |",
             "|---|---|" + "---|" * len(scen) + "---|"]
    for i, m in enumerate(ranking, 1):
        lines.append(f"| {i} | `{m}` | " + " | ".join(f"{table[m][s]:.2f}" for s in scen)
                     + f" | **{avg[m]:.3f}** |")
    md = "\n".join(lines)
    open(os.path.join(a.out, "cer_results.md"), "w").write(md + "\n")

    # Grafik
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vals = [avg[m] for m in ranking]
    bars = ax.barh(ranking[::-1], vals[::-1],
                   color=["#2e7d32" if m == ranking[0] else "#90a4ae" for m in ranking[::-1]])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center")
    ax.set_xlabel("Rata-rata CER (semakin kecil semakin baik)")
    ax.set_title("Perbandingan metode enhancement berdasarkan CER")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "cer_chart.png"), dpi=140)

    print(md)
    print(f"\nGround truth : {truth}")
    print(f"Metode terbaik (CER terendah): {ranking[0]}  (mean CER = {avg[ranking[0]]:.3f})")


if __name__ == "__main__":
    main()
