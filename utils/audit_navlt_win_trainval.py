#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import pickle
from pathlib import Path

ROOT = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win")
LOGS = ROOT / "navsim_logs" / "test"
BLOBS = ROOT / "sensor_blobs" / "test"

CAM = "CAM_F0"
EXPECTED = 14

def main():
    pkls = sorted(LOGS.glob("*.pkl"))
    print("num test pkls:", len(pkls))

    bad = []
    total_missing = 0

    for p in pkls:
        token = p.stem
        frames = pickle.load(open(p, "rb"))
        # 统计 pkl 中实际帧数
        n_frames = len(frames)

        # 收集该窗口期望的 CAM_F0 文件（相对 BLOBS/test 的路径）
        expected_paths = []
        for fr in frames:
            cams = fr.get("cams", {})
            if CAM not in cams:
                expected_paths.append(None)
                continue
            rel = cams[CAM].get("data_path")
            expected_paths.append(rel)

        # 检查存在性
        miss = 0
        miss_list = []
        for rel in expected_paths:
            if not rel:
                miss += 1
                miss_list.append("(missing_cam_key)")
                continue
            fp = BLOBS / rel
            if not fp.exists():
                miss += 1
                miss_list.append(str(rel))

        if miss > 0 or n_frames != EXPECTED:
            bad.append((token, n_frames, miss, ";".join(miss_list[:20])))
            total_missing += miss

    out_csv = ROOT / "audit_test_missing.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "num_frames_in_pkl", "num_missing_cam_files", "missing_examples(first20)"])
        w.writerows(bad)

    print("tokens with any issue:", len(bad))
    print("total missing cam files:", total_missing)
    print("saved:", out_csv)

if __name__ == "__main__":
    main()
