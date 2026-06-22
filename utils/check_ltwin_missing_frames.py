#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

ROOT = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win/sensor_blobs/trainval")
EXPECTED = 14

# 你要检查哪个相机目录：CAM_F0 / CAM_B0 / CAM_L0 ...
CAM = "CAM_F0"

# 允许的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def count_images(cam_dir: Path) -> int:
    if not cam_dir.exists() or not cam_dir.is_dir():
        return 0
    n = 0
    for p in cam_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            n += 1
    return n

def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT not found: {ROOT}")

    bad = []
    total = 0

    # token 目录就是 trainval/<token>/
    for token_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        total += 1
        token = token_dir.name
        cam_dir = token_dir / CAM
        n = count_images(cam_dir)

        if n < EXPECTED:
            bad.append((token, n, str(cam_dir)))

    out_txt = ROOT.parent / f"missing_{CAM}_{EXPECTED}_trainval.txt"
    out_csv = ROOT.parent / f"missing_{CAM}_{EXPECTED}_trainval.csv"

    out_txt.write_text("\n".join([f"{t}\t{n}\t{d}" for t, n, d in bad]) + ("\n" if bad else ""), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "num_frames_found", "cam_dir"])
        w.writerows(bad)

    print(f"Scanned token dirs: {total}")
    print(f"Tokens with <{EXPECTED} frames in {CAM}: {len(bad)}")
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
