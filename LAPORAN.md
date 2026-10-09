# Laporan Singkat — Metode Pipeline Verifikasi Ijazah

Dewa Raditya Saskara — F1G124061

## 1. Tujuan
Membaca nomor ijazah dengan OCR dan menentukan ada/tidaknya tanda tangan pejabat penandatangan (Rektor/Dekan) pada citra ijazah.

## 2. Metode yang Digunakan

### 2.1 Pra-pemrosesan
| Tahap | Metode | Alasan |
|---|---|---|
| Auto-orientasi | Coba rotasi 0/90/180/270°, pilih yang skor confidence OCR-nya tertinggi | Citra sample ter-scan terputar 90°; OCR gagal pada teks miring |
| Grayscale | `cv2.cvtColor(BGR2GRAY)` | Informasi yang dibutuhkan (tinta gelap vs kertas) ada pada intensitas; mengurangi kompleksitas |
| Enhancement global | Normalisasi iluminasi (bagi citra dengan estimasi background hasil *morphological closing* + Gaussian blur) lalu CLAHE | Meratakan warna kuning/tekstur guilloche kertas sehingga tinta menonjol |
| Pemisahan area | Crop ROI relatif (nomor, Rektor, Dekan) | Layout ijazah tetap, sehingga ROI relatif tahan terhadap perbedaan resolusi |

### 2.2 Jalur Nomor Ijazah
1. **Enhancement ROI** — 9 metode dibandingkan (lihat bagian 3); default = *adaptive Gaussian thresholding*.
2. **OCR** — Tesseract (`--oem 3`, `--psm 7` lalu fallback `--psm 6`), ditambah ekstraksi regex `\d{10,}` dan koreksi karakter yang sering tertukar (O→0, I/l→1, S→5, B→8).

### 2.3 Jalur Tanda Tangan
1. **Thresholding** — ambang = `min(Otsu, median_background − 55)`. Otsu mengikuti distribusi tinta; batas kedua mencegah Otsu memecah tekstur kertas ketika ROI sebenarnya kosong (menekan *false positive*).
2. **Morphology** — *opening* (ellipse 2×2) membuang bintik noise, *closing* (ellipse 9×9) menyambung goresan putus-putus.
3. **Signature Detection** — *connected components*; PRESENT bila komponen terbesar punya bounding box ≥ 30% lebar dan ≥ 18% tinggi ROI serta rasio tinta ≥ 1,2%. Alasan: tanda tangan berupa goresan kontinu yang besar, sedangkan teks cetak berupa komponen kecil-kecil.
4. **Keputusan akhir** — PRESENT jika minimal satu tanda tangan pejabat terdeteksi (`--require all` untuk mewajibkan keduanya).

### 2.4 Hasil pada sample
| Pengujian | Nomor Ijazah | Tanda Tangan |
|---|---|---|
| `ijazah_001.jpg` (asli) | 571012022000056 (CER 0) | PRESENT (Rektor ✔, Dekan ✔) |
| `ijazah_hanya_dekan.jpg` (TTD Rektor dihapus sintetis) | 571012022000056 | PRESENT (Rektor ✘, Dekan ✔) |
| `ijazah_tanpa_ttd.jpg` (kedua TTD dihapus sintetis) | 571012022000056 | **ABSENT** |

## 3. Analisis Metode Enhancement Berdasarkan CER

CER = (S + D + I) / N, dengan S = substitusi, D = penghapusan, I = penyisipan, N = panjang ground truth (15 digit). Semakin kecil semakin baik; nilai 1,00 umumnya berarti OCR tidak menemukan deret angka yang valid.

Karena hanya tersedia satu citra bersih (semua metode selain *histogram equalization* sudah CER 0 pada citra bersih), ROI nomor diuji pada 9 kondisi (bersih, blur, noise Gaussian, kontras rendah, iluminasi tidak merata, resolusi rendah, JPEG kualitas 6, salt-and-pepper, dan kombinasi), masing-masing dirata-ratakan pada 5 seed.

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

### Kesimpulan
- **Metode paling efektif: Adaptive Gaussian Thresholding** (rata-rata CER **0,084**), terendah dari 9 metode. Keunggulan utamanya ada pada **iluminasi tidak merata** (CER 0,00 sementara Otsu 1,00 dan CLAHE 1,00) karena ambangnya dihitung lokal per jendela, sedangkan Otsu memakai satu ambang global.
- **Baseline (grayscale saja)** peringkat ke-2 (0,135): Tesseract sudah memiliki binarisasi internal sehingga untuk teks cetak yang bersih enhancement tambahan tidak selalu membantu.
- **Histogram Equalization** terburuk (0,773): meratakan histogram global memperkuat tekstur latar sehingga merusak karakter, bahkan pada citra bersih (CER 0,27).
- **CLAHE dan upscale+CLAHE+sharpen** bagus pada citra bersih/blur/low-res, tetapi **memperkuat noise** (CER 0,84–1,00 pada noise) — peningkatan kontras lokal ikut menaikkan noise.
- Kelemahan adaptive threshold: pada **noise Gaussian kuat** CER masih 0,40 (Otsu dan baseline lebih baik di kondisi itu). Jadi tidak ada metode yang terbaik di semua kondisi; adaptive threshold dipilih karena **paling stabil secara rata-rata**.

### Keterbatasan evaluasi
Ukuran sampel kecil (satu citra, satu nomor ijazah, degradasi sintetis), sehingga peringkat bersifat indikatif. Untuk klaim yang lebih kuat diperlukan dataset ijazah/scan sungguhan yang lebih banyak.
