#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, pickle, os
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

NAVLT = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win")
META_DIR = NAVLT / "meta" / "trainval"
WIN_LOGS = NAVLT / "navsim_logs" / "trainval"
WIN_BLOBS = NAVLT / "sensor_blobs" / "trainval"

TOKEN2RAW = NAVLT / "token2rawscene_trainval.json"

RAW_BLOBS = Path("/path/to/ReCogDrive/navsim/dataset/sensor_blobs")
RAW_SPLITS = ["trainval", "test"]  # fallback

LINK_LIDAR = True
LINK_OCC = True
LINK_FLOW = True

# 并行进程数：建议 = 8~32（看你磁盘/CPU）
N_WORKERS = 16

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def fast_symlink(src: Path, dst: Path) -> bool:
    """不做 exists 预检查，直接 symlink，存在就跳过，快"""
    ensure_dir(dst.parent)
    try:
        os.symlink(src, dst)
        return True
    except FileExistsError:
        return False
    except OSError:
        return False

def find_src(scene_folder: str, subdir: str, filename: str) -> Optional[Path]:
    for sp in RAW_SPLITS:
        p = RAW_BLOBS / sp / scene_folder / subdir / filename
        if p.exists():
            return p
    return None

def handle_token(token: str, meta_scene_folder: str, raw_scene_name: Optional[str]) -> Tuple[int,int,int]:
    """return (pkl_changed, links_created, src_missing)"""
    pkl_path = WIN_LOGS / f"{token}.pkl"
    frames = pickle.load(open(pkl_path, "rb"))

    changed = 0
    links = 0
    missing = 0

    for fr in frames:
        # restore scene_name
        if raw_scene_name and fr.get("scene_name") != raw_scene_name:
            fr["scene_name"] = raw_scene_name
            changed = 1

        # cameras
        cams = fr.get("cams", {})
        if isinstance(cams, dict):
            for cam_name, cam_info in cams.items():
                if not (isinstance(cam_name, str) and cam_name.startswith("CAM_")):
                    continue
                if not isinstance(cam_info, dict):
                    continue
                rel = cam_info.get("data_path")
                if not isinstance(rel, str):
                    continue
                filename = Path(rel).name
                new_rel = f"{token}/{cam_name}/{filename}"
                if rel != new_rel:
                    cam_info["data_path"] = new_rel
                    changed = 1

                src = find_src(meta_scene_folder, cam_name, filename)
                if src is None:
                    missing += 1
                else:
                    dst = WIN_BLOBS / token / cam_name / filename
                    if fast_symlink(src, dst):
                        links += 1

        # lidar
        if LINK_LIDAR and isinstance(fr.get("lidar_path"), str) and fr["lidar_path"]:
            filename = Path(fr["lidar_path"]).name
            new_rel = f"{token}/MergedPointCloud/{filename}"
            if fr["lidar_path"] != new_rel:
                fr["lidar_path"] = new_rel
                changed = 1

            src = find_src(meta_scene_folder, "MergedPointCloud", filename)
            if src is None:
                missing += 1
            else:
                dst = WIN_BLOBS / token / "MergedPointCloud" / filename
                if fast_symlink(src, dst):
                    links += 1

        # occ / flow（如果你不需要 GT，直接关掉可再提速）
        if LINK_OCC and isinstance(fr.get("occ_gt_final_path"), str) and fr["occ_gt_final_path"]:
            filename = Path(fr["occ_gt_final_path"]).name
            subdir = "occ_gt_final_path"
            new_rel = f"{token}/{subdir}/{filename}"
            if fr["occ_gt_final_path"] != new_rel:
                fr["occ_gt_final_path"] = new_rel
                changed = 1
            src = find_src(meta_scene_folder, subdir, filename)
            if src is None:
                missing += 1
            else:
                dst = WIN_BLOBS / token / subdir / filename
                if fast_symlink(src, dst):
                    links += 1

        if LINK_FLOW and isinstance(fr.get("flow_gt_final_path"), str) and fr["flow_gt_final_path"]:
            filename = Path(fr["flow_gt_final_path"]).name
            subdir = "flow_gt_final_path"
            new_rel = f"{token}/{subdir}/{filename}"
            if fr["flow_gt_final_path"] != new_rel:
                fr["flow_gt_final_path"] = new_rel
                changed = 1
            src = find_src(meta_scene_folder, subdir, filename)
            if src is None:
                missing += 1
            else:
                dst = WIN_BLOBS / token / subdir / filename
                if fast_symlink(src, dst):
                    links += 1

    if changed:
        with open(pkl_path, "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)

    return changed, links, missing

def main():
    token2raw = json.loads(TOKEN2RAW.read_text(encoding="utf-8"))
    meta_files = sorted(META_DIR.glob("*.json"))
    print("meta files:", len(meta_files))
    print("token2raw size:", len(token2raw))

    jobs = []
    for mf in meta_files:
        token = mf.stem
        meta = json.loads(mf.read_text(encoding="utf-8"))
        meta_scene_folder = meta["scene_name"]
        raw_scene_name = token2raw.get(token)  # could be None
        jobs.append((token, meta_scene_folder, raw_scene_name))

    changed_pkls = 0
    links_created = 0
    src_missing = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(handle_token, *j) for j in jobs]
        for fut in as_completed(futs):
            c, l, m = fut.result()
            changed_pkls += c
            links_created += l
            src_missing += m

    print("changed pkls:", changed_pkls)
    print("links created:", links_created)
    print("missing src files:", src_missing)
    print("done")

if __name__ == "__main__":
    main()
