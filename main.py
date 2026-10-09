#!/usr/bin/env python3
"""CLI: verifikasi ijazah (OCR nomor + deteksi tanda tangan).

Contoh:
    python main.py data/ijazah_001.jpg
    python main.py data/ijazah_001.jpg --method clahe --out output --require all
"""
import argparse
import json
import sys

from src.enhancement import METHODS, DEFAULT_METHOD
from src.pipeline import verify


def main():
    ap = argparse.ArgumentParser(description="Verifikasi Ijazah: OCR + Signature Detection")
    ap.add_argument("image", help="path citra ijazah (jpg/png)")
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=list(METHODS),
                    help=f"metode enhancement ROI nomor (default: {DEFAULT_METHOD})")
    ap.add_argument("--require", default="any", choices=["any", "all"],
                    help="TTD PRESENT jika salah satu (any) / keduanya (all) pejabat terdeteksi")
    ap.add_argument("--out", default="output", help="folder output gambar tiap tahap")
    ap.add_argument("--json", action="store_true", help="cetak hasil detail dalam JSON")
    a = ap.parse_args()

    r = verify(a.image, out_dir=a.out, method=a.method, require=a.require)
    print(f"Nomor Ijazah : {r['nomor_ijazah']}")
    print(f"Tanda Tangan : {r['tanda_tangan']}")
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
