#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

NAVLT = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win")

META_DIR = NAVLT / "meta" / "trainval"
WIN_LOGS = NAVLT / "navsim_logs" / "trainval"
WIN_BLOBS = NAVLT / "sensor_blobs" / "trainval"

RAW_LOGS = Path("/path/to/ReCogDrive/navsim/dataset/navsim_logs/trainval")
RAW_BLOBS = Path("/path/to/ReCogDrive/navsim/dataset/sensor_blobs")
RAW_SPLITS = ["trainval", "test"]  # 兜底

ONLY_CAM = "CAM_F0"
LINK_LIDAR = True

N_WORKERS = 16  # 8~32都可以，I/O为主

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def symlink_replace(src: Path, dst: Path) -> bool:
    """dst 若是 symlink（含坏链）则替换；若是普通文件存在则跳过。"""
    ensure_dir(dst.parent)
    try:
        if dst.is_symlink():
            dst.unlink()
            os.symlink(src, dst)
            return True
        if dst.exists():
            return False
        os.symlink(src, dst)
        return True
    except OSError:
        return False

def find_blob(scene_folder: str, subdir: str, filename: str) -> Path | None:
    for sp in RAW_SPLITS:
        p = RAW_BLOBS / sp / scene_folder / subdir / filename
        if p.exists():
            return p
    return None

def process_scene(scene_folder: str, tokens: list[str]):
    """
    对一个 raw scene_folder（对应一个 raw pkl）：
      - 建 raw_token -> (raw_scene_name, cam_fname, lidar_fname) 映射
      - 逐 token 修窗口 pkl（14帧逐帧回填）
      - 为每帧创建 symlink
    返回统计信息
    """
    raw_pkl = RAW_LOGS / f"{scene_folder}.pkl"
    if not raw_pkl.exists():
        return (scene_folder, 0, 0, len(tokens), 0)  # changed, links, miss_raw_scene, miss_blob

    raw_frames = pickle.load(open(raw_pkl, "rb"))

    raw_map = {}
    for fr in raw_frames:
        tk = fr.get("token")
        if not tk:
            continue
        cam_rel = fr.get("cams", {}).get(ONLY_CAM, {}).get("data_path", "")
        cam_fname = Path(cam_rel).name if isinstance(cam_rel, str) and cam_rel else ""
        lidar_rel = fr.get("lidar_path", "")
        lidar_fname = Path(lidar_rel).name if isinstance(lidar_rel, str) and lidar_rel else ""
        raw_map[tk] = (fr.get("scene_name"), cam_fname, lidar_fname)

    changed_pkls = 0
    links = 0
    miss_blob = 0
    miss_raw_scene = 0

    for token in tokens:
        win_pkl = WIN_LOGS / f"{token}.pkl"
        if not win_pkl.exists():
            continue

        frames = pickle.load(open(win_pkl, "rb"))
        changed = False

        # 收集窗口内需要的文件集合（去重减少 symlink 次数）
        need_cam_files = set()
        need_lidar_files = set()

        for fr in frames:
            ftok = fr.get("token")  # 这是“帧 token”，不是窗口 token
            if ftok not in raw_map:
                miss_raw_scene += 1
                continue

            raw_scene_name, cam_fname, lidar_fname = raw_map[ftok]

            # 恢复 scene_name（用 raw 的 log-xxxx-scene-xxxx）
            if raw_scene_name and fr.get("scene_name") != raw_scene_name:
                fr["scene_name"] = raw_scene_name
                changed = True

            # CAM_F0：逐帧回填真实文件名
            cam_rel_new = f"{token}/{ONLY_CAM}/{cam_fname}"
            cam_info = fr.get("cams", {}).get(ONLY_CAM, None)
            if isinstance(cam_info, dict):
                if cam_info.get("data_path") != cam_rel_new:
                    cam_info["data_path"] = cam_rel_new
                    changed = True
                if cam_fname:
                    need_cam_files.add(cam_fname)

            # lidar：逐帧回填真实文件名（可选）
            if LINK_LIDAR and lidar_fname:
                lidar_rel_new = f"{token}/MergedPointCloud/{lidar_fname}"
                if fr.get("lidar_path") != lidar_rel_new:
                    fr["lidar_path"] = lidar_rel_new
                    changed = True
                need_lidar_files.add(lidar_fname)

        if changed:
            with open(win_pkl, "wb") as f:
                pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
            changed_pkls += 1

        # 逐文件创建 symlink（目标按 token 组织）
        for cam_fname in need_cam_files:
            src = find_blob(scene_folder, ONLY_CAM, cam_fname)
            if src is None:
                miss_blob += 1
                continue
            dst = WIN_BLOBS / token / ONLY_CAM / cam_fname
            if symlink_replace(src, dst):
                links += 1

        if LINK_LIDAR:
            for lidar_fname in need_lidar_files:
                src = find_blob(scene_folder, "MergedPointCloud", lidar_fname)
                if src is None:
                    miss_blob += 1
                    continue
                dst = WIN_BLOBS / token / "MergedPointCloud" / lidar_fname
                if symlink_replace(src, dst):
                    links += 1

    return (scene_folder, changed_pkls, links, miss_raw_scene, miss_blob)

def main():
    meta_files = sorted(META_DIR.glob("*.json"))
    print("meta files:", len(meta_files))

    scene2tokens = defaultdict(list)
    for mf in meta_files:
        token = mf.stem
        pkl = WIN_LOGS / f"{token}.pkl"
        if not pkl.exists():
            continue
        meta = json.loads(mf.read_text(encoding="utf-8"))
        scene_folder = meta["scene_name"]  # 2021...veh...
        scene2tokens[scene_folder].append(token)

    scenes = list(scene2tokens.items())
    print("unique scenes:", len(scenes))
    print("ONLY_CAM:", ONLY_CAM, "LINK_LIDAR:", LINK_LIDAR)

    total_changed = 0
    total_links = 0
    total_miss_raw = 0
    total_miss_blob = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(process_scene, scene, toks) for scene, toks in scenes]
        for fut in as_completed(futs):
            scene, c, l, mr, mb = fut.result()
            total_changed += c
            total_links += l
            total_miss_raw += mr
            total_miss_blob += mb

    print("changed pkls:", total_changed)
    print("links created/replaced:", total_links)
    print("missing raw frame tokens:", total_miss_raw)
    print("missing blob files:", total_miss_blob)
    print("done")

if __name__ == "__main__":
    main()
