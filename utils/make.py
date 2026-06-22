#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re
from pathlib import Path

# ====== 你只需要确认这三个路径 ======
SCENES_FILE = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win/missing_scenes_from_bad_tokens.txt")
MAP_JSON    = Path("/path/to/ReCogDrive/_openscene_map/openscene_sensor_trainval_0-199.json")
OUT_DIR     = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win")
# =====================================

NAVLT_SUP = Path("/path/to/ReCogDrive/navsim/dataset/navlt_sup")

BASE = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1"
CAM_DIR = "openscene_sensor_trainval_camera"
LID_DIR = "openscene_sensor_trainval_lidar"

KEY_RE = re.compile(r"^openscene_sensor_trainval_(\d+)$")

def main():
    scenes = {s.strip() for s in SCENES_FILE.read_text().splitlines() if s.strip()}
    mapping = json.loads(MAP_JSON.read_text())

    idx2logs = {}
    for k, v in mapping.items():
        if not isinstance(v, list):
            continue
        m = KEY_RE.match(str(k))
        if not m:
            continue
        idx = int(m.group(1))
        idx2logs[idx] = set(v)

    required = sorted([idx for idx, logs in idx2logs.items() if scenes.intersection(logs)])
    covered = sum(len(scenes.intersection(idx2logs[i])) for i in required)

    # 输出索引与文件名清单
    out_idx = OUT_DIR / "required_trainval_shard_indices.txt"
    out_cam = OUT_DIR / "required_trainval_camera_tgz.txt"
    out_lid = OUT_DIR / "required_trainval_lidar_tgz.txt"

    out_idx.write_text("\n".join(map(str, required)) + "\n", encoding="utf-8")
    out_cam.write_text("\n".join([f"openscene_sensor_trainval_camera_{i}.tgz" for i in required]) + "\n", encoding="utf-8")
    out_lid.write_text("\n".join([f"openscene_sensor_trainval_lidar_{i}.tgz"  for i in required]) + "\n", encoding="utf-8")

    # 生成下载脚本（下载到 navlt_sup）
    sh_cam = OUT_DIR / "download_required_trainval_camera_to_navlt_sup.sh"
    sh_lid = OUT_DIR / "download_required_trainval_lidar_to_navlt_sup.sh"

    cam_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'mkdir -p "{NAVLT_SUP}/openscene_trainval_camera" && cd "{NAVLT_SUP}/openscene_trainval_camera"',
        f'echo "Downloading {len(required)} camera shards to {NAVLT_SUP}/openscene_trainval_camera"',
    ]
    for j, i in enumerate(required, 1):
        tgz = f"openscene_sensor_trainval_camera_{i}.tgz"
        url = f"{BASE}/{CAM_DIR}/{tgz}"
        cam_lines.append(f'echo "[{j}/{len(required)}] {tgz}"')
        cam_lines.append(f'aria2c -c -x 8 -s 8 -k 1M -o "{tgz}" "{url}"')

    lid_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'mkdir -p "{NAVLT_SUP}/openscene_trainval_lidar" && cd "{NAVLT_SUP}/openscene_trainval_lidar"',
        f'echo "Downloading {len(required)} lidar shards to {NAVLT_SUP}/openscene_trainval_lidar"',
    ]
    for j, i in enumerate(required, 1):
        tgz = f"openscene_sensor_trainval_lidar_{i}.tgz"
        url = f"{BASE}/{LID_DIR}/{tgz}"
        lid_lines.append(f'echo "[{j}/{len(required)}] {tgz}"')
        lid_lines.append(f'aria2c -c -x 8 -s 8 -k 1M -o "{tgz}" "{url}"')

    sh_cam.write_text("\n".join(cam_lines) + "\n", encoding="utf-8")
    sh_lid.write_text("\n".join(lid_lines) + "\n", encoding="utf-8")
    sh_cam.chmod(0o755)
    sh_lid.chmod(0o755)

    print("scenes:", len(scenes))
    print("mapping shards parsed:", len(idx2logs))
    print("required shard indices:", len(required))
    print("covered scenes (upper bound):", covered)
    print("wrote:", out_idx)
    print("wrote:", out_cam)
    print("wrote:", out_lid)
    print("wrote:", sh_cam)
    print("wrote:", sh_lid)

if __name__ == "__main__":
    main()

