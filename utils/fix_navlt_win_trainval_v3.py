#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, pickle, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

NAVLT = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win")

META_DIR = NAVLT / "meta" / "trainval"
OUT_LOGS = NAVLT / "navsim_logs" / "trainval"
OUT_BLOBS = NAVLT / "sensor_blobs" / "trainval"

RAW_LOGS = Path("/path/to/ReCogDrive/navsim/dataset/navsim_logs/trainval")
RAW_BLOBS = Path("/path/to/ReCogDrive/navsim/dataset/sensor_blobs/trainval")

ONLY_CAM = "CAM_F0"
LINK_LIDAR = True

# 并行进程数（I/O为主）
N_WORKERS = 16

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def symlink_replace(src: Path, dst: Path) -> bool:
    ensure_dir(dst.parent)
    try:
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            return False
        os.symlink(src, dst)
        return True
    except OSError:
        return False

def rebuild_one(meta_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token = meta["anchor_token"]          # 目标窗口token（也是文件名）
    scene = meta["scene_name"]            # raw scene_folder
    s = meta["window_start_frame_index_in_scene"]
    e = meta["window_end_frame_index_in_scene"]  # inclusive

    raw_pkl = RAW_LOGS / f"{scene}.pkl"
    if not raw_pkl.exists():
        return (0, 0, 1, 0)  # rebuilt, links, miss_raw_pkl, miss_blob

    raw_frames = pickle.load(open(raw_pkl, "rb"))
    if e >= len(raw_frames) or s < 0 or s > e:
        return (0, 0, 0, 1)  # index bad 记到 miss_blob 里方便统计

    win = raw_frames[s:e+1]
    if len(win) != 14:
        # 理论上就是14，若不是也返回
        return (0, 0, 0, 1)

    # 重建窗口 frames：只改路径到 token 前缀；scene_name 保持 raw 的 log-xxxx-scene-xxxx（更语义正确）
    links = 0
    miss_blob = 0

    for fr in win:
        # CAM_F0
        cam_rel = fr["cams"][ONLY_CAM]["data_path"]
        cam_fname = Path(cam_rel).name
        fr["cams"][ONLY_CAM]["data_path"] = f"{token}/{ONLY_CAM}/{cam_fname}"

        src_cam = RAW_BLOBS / scene / ONLY_CAM / cam_fname
        if not src_cam.exists():
            miss_blob += 1
        else:
            dst_cam = OUT_BLOBS / token / ONLY_CAM / cam_fname
            if symlink_replace(src_cam, dst_cam):
                links += 1

        # lidar
        if LINK_LIDAR:
            lidar_rel = fr.get("lidar_path", "")
            if isinstance(lidar_rel, str) and lidar_rel:
                lidar_fname = Path(lidar_rel).name
                fr["lidar_path"] = f"{token}/MergedPointCloud/{lidar_fname}"
                src_lidar = RAW_BLOBS / scene / "MergedPointCloud" / lidar_fname
                if not src_lidar.exists():
                    miss_blob += 1
                else:
                    dst_lidar = OUT_BLOBS / token / "MergedPointCloud" / lidar_fname
                    if symlink_replace(src_lidar, dst_lidar):
                        links += 1

    # 输出窗口 pkl（覆盖写）
    ensure_dir(OUT_LOGS)
    out_pkl = OUT_LOGS / f"{token}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(win, f, protocol=pickle.HIGHEST_PROTOCOL)

    return (1, links, 0, miss_blob)

def main():
    meta_files = sorted(META_DIR.glob("*.json"))
    print("meta files:", len(meta_files))
    rebuilt = 0
    links = 0
    miss_raw = 0
    miss_blob = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(rebuild_one, mf) for mf in meta_files]
        for fut in as_completed(futs):
            r, l, mr, mb = fut.result()
            rebuilt += r
            links += l
            miss_raw += mr
            miss_blob += mb

    print("rebuilt pkls:", rebuilt)
    print("links created/replaced:", links)
    print("missing raw pkls:", miss_raw)
    print("missing blobs or bad indices:", miss_blob)
    print("done")

if __name__ == "__main__":
    main()
