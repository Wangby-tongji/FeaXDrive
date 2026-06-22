#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# ====== paths you gave ======
NAVLT = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win")
META_DIR = NAVLT / "meta" / "trainval"
OUT_LOGS = NAVLT / "navsim_logs" / "trainval"
OUT_BLOBS = NAVLT / "sensor_blobs" / "trainval"

RAW_LOGS = Path("/path/to/ReCogDrive/navsim/dataset/navsim_logs/trainval")
RAW_BLOBS_ROOT = Path("/path/to/ReCogDrive/navsim/dataset/sensor_blobs")
RAW_BLOBS_SPLITS = ["trainval", "test"]  # fallback

# ====== knobs ======
N_WORKERS = 16            # 建议 8~32，I/O 大的话不要太夸张
INCLUDE_LIDAR = True
INCLUDE_OCC = True        # 如果你不需要 occ/flow，可以关掉更快
INCLUDE_FLOW = True
RECREATE_OUTPUT_DIRS = True  # True: 先删再建；False: 增量修复

EXPECTED_FRAMES = 14      # 4 history + 10 future

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def atomic_pickle_dump(obj, dst: Path):
    """Write pickle atomically to avoid truncated files."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(dst)

def fast_symlink(src: Path, dst: Path) -> bool:
    """Try create symlink without prechecking; skip if exists."""
    ensure_dir(dst.parent)
    try:
        os.symlink(src, dst)
        return True
    except FileExistsError:
        return False
    except OSError:
        return False

def find_src_blob(scene_folder: str, subdir: str, filename: str) -> Optional[Path]:
    """Locate source sensor file in sensor_blobs/(trainval|test)/scene_folder/subdir/filename"""
    for sp in RAW_BLOBS_SPLITS:
        p = RAW_BLOBS_ROOT / sp / scene_folder / subdir / filename
        if p.exists():
            return p
    return None

def load_raw_scene_and_window(scene_folder: str, start: int, end: int) -> Tuple[List[dict], Optional[str]]:
    """
    Load raw frames from navsim_logs/trainval/{scene_folder}.pkl,
    slice out the window (inclusive or exclusive auto-detect),
    and return (window_frames, raw_scene_name 'log-xxxx-scene-xxxx' from anchor frame).
    """
    raw_pkl = RAW_LOGS / f"{scene_folder}.pkl"
    frames = pickle.load(open(raw_pkl, "rb"))

    # try inclusive first
    w1 = frames[start:end + 1]
    if len(w1) == EXPECTED_FRAMES:
        window = w1
    else:
        w2 = frames[start:end]
        if len(w2) == EXPECTED_FRAMES:
            window = w2
        else:
            raise ValueError(f"Bad window len for {scene_folder} start={start} end={end}: got {len(w1)} or {len(w2)}")

    # anchor index = num_history_frames-1 = 3
    anchor = window[3]
    raw_scene_name = anchor.get("scene_name")
    return window, raw_scene_name

def fix_window_frames(token: str, window: List[dict], raw_scene_name: Optional[str]) -> List[dict]:
    """
    Make window self-consistent for NAVSIM pipeline:
      - log_name -> token (so file stem == log_name)
      - scene_name -> raw_scene_name (keep official semantics)
      - sample_prev/sample_next consistent within this window
      - cams[*].data_path rewritten to token/<CAM>/<filename>
      - lidar/occ/flow paths rewritten to token/<subdir>/<filename>
    """
    # gather frame tokens in this window
    frame_tokens = [fr.get("token") for fr in window]

    for i, fr in enumerate(window):
        # keep original tokens/timestamps/map_location/etc
        fr["log_name"] = token
        if raw_scene_name is not None:
            fr["scene_name"] = raw_scene_name

        # prev/next
        fr["sample_prev"] = frame_tokens[i - 1] if i > 0 else None
        fr["sample_next"] = frame_tokens[i + 1] if i < len(window) - 1 else None

        # cameras
        cams = fr.get("cams", {})
        if isinstance(cams, dict):
            for cam_name, cam_info in cams.items():
                if not (isinstance(cam_name, str) and cam_name.startswith("CAM_")):
                    continue
                if not isinstance(cam_info, dict):
                    continue
                rel = cam_info.get("data_path")
                if isinstance(rel, str) and rel:
                    filename = Path(rel).name
                    cam_info["data_path"] = f"{token}/{cam_name}/{filename}"

        # lidar
        if INCLUDE_LIDAR and isinstance(fr.get("lidar_path"), str) and fr["lidar_path"]:
            filename = Path(fr["lidar_path"]).name
            fr["lidar_path"] = f"{token}/MergedPointCloud/{filename}"

        # occ/flow
        if INCLUDE_OCC and isinstance(fr.get("occ_gt_final_path"), str) and fr["occ_gt_final_path"]:
            filename = Path(fr["occ_gt_final_path"]).name
            fr["occ_gt_final_path"] = f"{token}/occ_gt_final_path/{filename}"

        if INCLUDE_FLOW and isinstance(fr.get("flow_gt_final_path"), str) and fr["flow_gt_final_path"]:
            filename = Path(fr["flow_gt_final_path"]).name
            fr["flow_gt_final_path"] = f"{token}/flow_gt_final_path/{filename}"

    return window

def link_sensor_files(token: str, scene_folder: str, window: List[dict]) -> Tuple[int, int]:
    """
    Create symlinks for everything referenced by rewritten paths.
    Return (links_created, missing_src).
    """
    links = 0
    missing = 0

    for fr in window:
        # cameras
        cams = fr.get("cams", {})
        if isinstance(cams, dict):
            for cam_name, cam_info in cams.items():
                if not (isinstance(cam_name, str) and cam_name.startswith("CAM_")):
                    continue
                if not isinstance(cam_info, dict):
                    continue
                rel = cam_info.get("data_path")
                if not isinstance(rel, str) or not rel:
                    continue
                filename = Path(rel).name
                src = find_src_blob(scene_folder, cam_name, filename)
                if src is None:
                    missing += 1
                else:
                    dst = OUT_BLOBS / token / cam_name / filename
                    if fast_symlink(src, dst):
                        links += 1

        # lidar
        if INCLUDE_LIDAR and isinstance(fr.get("lidar_path"), str) and fr["lidar_path"]:
            filename = Path(fr["lidar_path"]).name
            src = find_src_blob(scene_folder, "MergedPointCloud", filename)
            if src is None:
                missing += 1
            else:
                dst = OUT_BLOBS / token / "MergedPointCloud" / filename
                if fast_symlink(src, dst):
                    links += 1

        # occ
        if INCLUDE_OCC and isinstance(fr.get("occ_gt_final_path"), str) and fr["occ_gt_final_path"]:
            filename = Path(fr["occ_gt_final_path"]).name
            src = find_src_blob(scene_folder, "occ_gt_final_path", filename)
            if src is None:
                missing += 1
            else:
                dst = OUT_BLOBS / token / "occ_gt_final_path" / filename
                if fast_symlink(src, dst):
                    links += 1

        # flow
        if INCLUDE_FLOW and isinstance(fr.get("flow_gt_final_path"), str) and fr["flow_gt_final_path"]:
            filename = Path(fr["flow_gt_final_path"]).name
            src = find_src_blob(scene_folder, "flow_gt_final_path", filename)
            if src is None:
                missing += 1
            else:
                dst = OUT_BLOBS / token / "flow_gt_final_path" / filename
                if fast_symlink(src, dst):
                    links += 1

    return links, missing

def process_one(meta_path: Path) -> Tuple[str, int, int, int]:
    """
    One token job.
    Return (token, pkl_written(0/1), links_created, missing_src)
    """
    token = meta_path.stem
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    scene_folder = meta["scene_name"]  # 2021....veh...
    start = meta["window_start_frame_index_in_scene"]
    end = meta["window_end_frame_index_in_scene"]

    # 1) load raw window + raw scene_name
    window, raw_scene_name = load_raw_scene_and_window(scene_folder, start, end)

    # 2) fix fields + rewrite paths
    window = fix_window_frames(token, window, raw_scene_name)

    # 3) write new token-level pkl atomically
    out_pkl = OUT_LOGS / f"{token}.pkl"
    ensure_dir(OUT_LOGS)
    atomic_pickle_dump(window, out_pkl)

    # 4) link sensor files
    ensure_dir(OUT_BLOBS / token)
    links, missing = link_sensor_files(token, scene_folder, window)

    return token, 1, links, missing

def main():
    if RECREATE_OUTPUT_DIRS:
        # remove and recreate outputs
        if OUT_LOGS.exists():
            shutil.rmtree(OUT_LOGS)
        if OUT_BLOBS.exists():
            shutil.rmtree(OUT_BLOBS)
        ensure_dir(OUT_LOGS)
        ensure_dir(OUT_BLOBS)

    metas = sorted(META_DIR.glob("*.json"))
    print("meta files:", len(metas))
    print("workers:", N_WORKERS)
    print("include lidar/occ/flow:", INCLUDE_LIDAR, INCLUDE_OCC, INCLUDE_FLOW)
    print("recreate outputs:", RECREATE_OUTPUT_DIRS)

    written = 0
    links_total = 0
    missing_total = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(process_one, mp) for mp in metas]
        for fut in as_completed(futs):
            try:
                _, w, l, m = fut.result()
                written += w
                links_total += l
                missing_total += m
            except Exception as e:
                failed += 1

    print("pkls written:", written, "/", len(metas))
    print("links created:", links_total)
    print("missing source files:", missing_total)
    print("failed jobs:", failed)
    print("done")

if __name__ == "__main__":
    main()
