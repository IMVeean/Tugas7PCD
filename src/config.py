"""Konfigurasi ROI (Region of Interest) dalam koordinat RELATIF (0..1).

Koordinat dihitung terhadap citra ijazah yang sudah berorientasi landscape
(x = kiri->kanan, y = atas->bawah), format: (x1, y1, x2, y2).
Layout ijazah UI relatif tetap, sehingga ROI relatif robust terhadap
perbedaan resolusi scan.
"""

# Area nomor ijazah ("Nomor ijazah: 5710...") di kiri-bawah
ROI_NOMOR = (0.03, 0.885, 0.34, 0.960)

# Area tanda tangan pejabat (di atas nama pejabat, di bawah jabatan)
ROI_TTD_REKTOR = (0.12, 0.700, 0.42, 0.835)
ROI_TTD_DEKAN = (0.62, 0.675, 0.90, 0.835)
# Tanda tangan pemilik ijazah (di bawah foto)
ROI_TTD_PEMILIK = (0.42, 0.835, 0.62, 0.925)

# Pola nomor ijazah: deretan angka panjang (>= 10 digit)
NOMOR_REGEX = r"\d{10,}"

# Ground-truth untuk evaluasi CER (sample ijazah_001.jpg)
GROUND_TRUTH = {"ijazah_001.jpg": "571012022000056"}
