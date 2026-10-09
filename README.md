# Mini Project Citra Digital — Verifikasi Ijazah (OCR + Deteksi Tanda Tangan)

**Nama:** La Ode Muhammad Febriansyah  **NIM:** F1G124037

Prototype yang menerima citra ijazah dan menghasilkan:

```
Input : data/ijazah_001.jpg
Output:
Nomor Ijazah : 571012022000056
Tanda Tangan : PRESENT
```

## Pipeline

```
Citra Ijazah → (auto-orientasi) → Grayscale → Image Enhancement (global)
        ┌───────────────────────────────┴───────────────────────────────┐
   Area Nomor Ijazah                                         Area Tanda Tangan
        ↓                                                               ↓
   Enhancement ROI                                                Thresholding
        ↓                                                               ↓
   OCR (Tesseract)                                                  Morphology
        ↓                                                               ↓
   Nomor Ijazah                                                Signature Detection
        └───────────────────────────────┬───────────────────────────────┘
                                 Hasil Verifikasi
```


## Cara Menjalankan (How to Run)

### 1. Prasyarat
- Python 3.9+
- **Tesseract OCR** terpasang di sistem:
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: unduh installer dari <https://github.com/UB-Mannheim/tesseract/wiki>, lalu
    tambahkan folder instalasi ke PATH (atau set
    `pytesseract.pytesseract.tesseract_cmd` di `src/ocr.py`).

### 2. Instal dependensi
```bash
git clone <URL-REPOSITORY-ANDA>
cd <nama-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Jalankan verifikasi
```bash
python main.py data/ijazah_001.jpg
# Nomor Ijazah : 571012022000056
# Tanda Tangan : PRESENT
```

### 4. Evaluasi CER antar metode enhancement
```bash
python evaluate.py          # menghasilkan output/cer_results.{csv,md} dan cer_chart.png
```

### 5. Unit test
```bash
python -m unittest discover tests -v
```

## Hasil Evaluasi CER (ringkas)

Ground truth: `571012022000056`. CER dirata-ratakan pada 9 kondisi degradasi × 5 seed.

| Peringkat | Metode | clean | blur | noise | low_contrast | uneven_light | low_res | jpeg_q6 | salt_pepper | kombinasi | **Rata-rata CER** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `adaptive_thresh` | 0.00 | 0.00 | 0.40 | 0.00 | 0.00 | 0.07 | 0.00 | 0.17 | 0.12 | **0.084** |
| 2 | `baseline_gray` | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 | 0.13 | 0.00 | 0.24 | 0.71 | **0.135** |
| 3 | `otsu` | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.07 | 0.00 | 0.24 | 0.32 | **0.181** |
| 4 | `unsharp` | 0.00 | 0.00 | 0.83 | 0.00 | 0.13 | 0.13 | 0.00 | 0.03 | 0.52 | **0.182** |
| 5 | `upscale_median_otsu` | 0.00 | 0.07 | 0.00 | 0.00 | 0.27 | 0.07 | 0.00 | 0.24 | 1.00 | **0.182** |
| 6 | `clahe` | 0.00 | 0.00 | 0.84 | 0.00 | 1.00 | 0.13 | 0.00 | 0.04 | 0.73 | **0.305** |
| 7 | `clahe_bilateral_otsu` | 0.00 | 0.07 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.23 | 0.84 | **0.348** |
| 8 | `upscale_clahe_sharpen` | 0.00 | 0.00 | 1.00 | 0.00 | 0.33 | 0.00 | 0.00 | 1.00 | 1.00 | **0.370** |
| 9 | `hist_equalization` | 0.27 | 1.00 | 0.43 | 1.00 | 0.33 | 1.00 | 1.00 | 0.93 | 1.00 | **0.773** |

**Metode paling efektif: `adaptive_thresh`** (rata-rata CER 0.084). Penjelasan lengkap di [LAPORAN.md](LAPORAN.md).
