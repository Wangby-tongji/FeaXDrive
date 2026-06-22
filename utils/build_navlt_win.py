#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_mapping(jsonl_path: Path) -> List[dict]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                rows.append(r)
    return rows


def copy_or_symlink(src: Path, dst: Path, use_symlink: bool) -> None:
    if dst.exists():
        return
    safe_mkdir(dst.parent)
    if use_symlink:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def extract_window(frames: List[dict], anchor_token: str, num_history: int, num_future: int) -> Optional[Tuple[int, int, List[dict]]]:
    """
    frames: full sequence (e.g., 804 dicts)
    anchor_token: hex token of anchor frame (history end)
    return (start_idx, end_idx, window_frames)
    window length = num_history + num_future
    window indices: [idx-(num_history-1), idx+num_future] inclusive
    """
    idx = None
    for i, fr in enumerate(frames):
        if fr.get("token") == anchor_token:
            idx = i
            break
    if idx is None:
        return None

    start = idx - (num_history - 1)
    end = idx + num_future
    if start < 0 or end >= len(frames):
        return None

    win = frames[start : end + 1]
    if len(win) != (num_history + num_future):
        return None
    return start, end, win


def rewrite_paths_and_collect_assets(
    window_frames: List[dict],
    split: str,
    anchor_token: str,
    src_sensor_root: Path,
    dst_sensor_root: Path,
    use_symlink: bool,
) -> Dict[str, int]:
    """
    For each frame in window:
      - rewrite cams[cam]['data_path'] to "<anchor_token>/<cam>/<filename>"
      - rewrite lidar_path (and occ/flow gt paths if present) to "<anchor_token>/<subdir>/<filename>"
      - create symlinks/copies in dst_sensor_root/split/<anchor_token>/...
    Returns counts.
    """
    dst_win_root = dst_sensor_root / split / anchor_token
    safe_mkdir(dst_win_root)

    counts = {"images": 0, "lidar": 0, "occ": 0, "flow": 0, "missing_assets": 0}

    def link_rel(rel: str, kind_subdir: Optional[str] = None) -> Optional[str]:
        """
        rel is original relative path like:
          2021..._00083_00485/CAM_F0/xxxx.jpg
          2021..._00083_00485/MergedPointCloud/xxxx.bin
        We link the file into:
          <anchor_token>/<CAM_F0>/xxxx.jpg   (for cams)
          <anchor_token>/<MergedPointCloud>/xxxx.bin  (for lidar/gt)
        And return the new relative path "<anchor_token>/<...>/filename".
        """
        try:
            rel = rel.strip().lstrip("/")
            src = src_sensor_root / split / rel
            if not src.exists():
                counts["missing_assets"] += 1
                return None
            # determine target subdir+filename
            parts = Path(rel).parts
            filename = parts[-1]
            if kind_subdir is not None:
                subdir = kind_subdir
            else:
                # use last two components' parent as subdir (CAM_F0 / MergedPointCloud / etc.)
                subdir = parts[-2] if len(parts) >= 2 else "UNKNOWN"

            dst = dst_win_root / subdir / filename
            copy_or_symlink(src, dst, use_symlink)
            return f"{anchor_token}/{subdir}/{filename}"
        except Exception:
            counts["missing_assets"] += 1
            return None

    # Process each frame
    for fr in window_frames:
        # cameras
        cams = fr.get("cams", {})
        for cam_name, cam_info in list(cams.items()):
            dp = cam_info.get("data_path")
            if not dp:
                continue
            new_rel = link_rel(dp, kind_subdir=cam_name)
            if new_rel is None:
                continue
            cam_info["data_path"] = new_rel
            counts["images"] += 1

        # lidar / merged point cloud
        if isinstance(fr.get("lidar_path"), str) and fr["lidar_path"]:
            new_rel = link_rel(fr["lidar_path"], kind_subdir="MergedPointCloud")
            if new_rel is not None:
                fr["lidar_path"] = new_rel
                counts["lidar"] += 1

        # optional GT assets (if present)
        for k, kk in [("occ_gt_final_path", "occ"), ("flow_gt_final_path", "flow")]:
            if isinstance(fr.get(k), str) and fr[k]:
                # put into subdir named by key for clarity
                new_rel = link_rel(fr[k], kind_subdir=k)
                if new_rel is not None:
                    fr[k] = new_rel
                    counts[kk] += 1

    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping_jsonl", type=str, required=True, help="lt_mapping.jsonl from your Impromptu->NAVSIM mapping step")
    ap.add_argument("--src_logs_test", type=str, required=True)
    ap.add_argument("--src_logs_trainval", type=str, required=True)
    ap.add_argument("--src_sensor_root", type=str, required=True, help=".../sensor_blobs (parent containing test/trainval)")
    ap.add_argument("--dst_root", type=str, required=True, help="Output root navlt_win")
    ap.add_argument("--num_history_frames", type=int, default=4)
    ap.add_argument("--num_future_frames", type=int, default=10)
    ap.add_argument("--use_symlink", action="store_true", help="Symlink assets instead of copying (recommended)")
    ap.add_argument("--splits", type=str, default="test,trainval", help="Comma-separated splits to build, default: test,trainval")
    args = ap.parse_args()

    mapping_jsonl = Path(args.mapping_jsonl).expanduser().resolve()
    src_logs = {
        "test": Path(args.src_logs_test).expanduser().resolve(),
        "trainval": Path(args.src_logs_trainval).expanduser().resolve(),
    }
    src_sensor_root = Path(args.src_sensor_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in ("test", "trainval"):
            raise ValueError(f"Unsupported split: {s}")

    # output structure
    dst_logs_root = dst_root / "navsim_logs"
    dst_sensor_root = dst_root / "sensor_blobs"
    dst_meta_root = dst_root / "meta"
    dst_tokcat_root = dst_root / "tokens_by_category"

    safe_mkdir(dst_logs_root)
    safe_mkdir(dst_sensor_root)
    safe_mkdir(dst_meta_root)
    safe_mkdir(dst_tokcat_root)

    rows = load_mapping(mapping_jsonl)

    # group tokens by category/split for later filtering
    tok_by_cat: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    summary = {
        "built": 0,
        "skipped_missing_scene_pkl": 0,
        "skipped_token_not_found_or_oob": 0,
        "asset_counts": {"images": 0, "lidar": 0, "occ": 0, "flow": 0, "missing_assets": 0},
    }

    # cache loaded pkls (scene_name -> frames) to avoid repeated loads
    cache: Dict[Tuple[str, str], List[dict]] = {}

    for r in rows:
        split = r.get("split")
        if split not in splits:
            continue

        category = r.get("category", "UNKNOWN")
        scene_name = r.get("scene_name")
        anchor_token = r.get("anchor_token")

        if not scene_name or not anchor_token:
            continue

        scene_pkl = src_logs[split] / f"{scene_name}.pkl"
        if not scene_pkl.exists():
            summary["skipped_missing_scene_pkl"] += 1
            continue

        key = (split, scene_name)
        if key not in cache:
            frames = pickle.load(open(scene_pkl, "rb"))
            if not isinstance(frames, list) or not frames:
                summary["skipped_missing_scene_pkl"] += 1
                continue
            cache[key] = frames

        frames = cache[key]
        win_info = extract_window(frames, anchor_token, args.num_history_frames, args.num_future_frames)
        if win_info is None:
            summary["skipped_token_not_found_or_oob"] += 1
            continue
        start_idx, end_idx, win_frames = win_info

        # make a deep-ish copy of window frames so we can rewrite paths without altering cached frames
        # (frames are dicts; shallow copy per frame is ok for our edits)
        win_frames_copy = []
        for fr in win_frames:
            fr2 = dict(fr)
            # copy nested dicts we modify
            if isinstance(fr2.get("cams"), dict):
                fr2["cams"] = {k: dict(v) for k, v in fr2["cams"].items()}
            win_frames_copy.append(fr2)

        # rewrite paths + link/copy assets into dst sensor folder
        cnts = rewrite_paths_and_collect_assets(
            win_frames_copy, split, anchor_token,
            src_sensor_root=src_sensor_root,
            dst_sensor_root=dst_sensor_root,
            use_symlink=args.use_symlink,
        )
        for k in summary["asset_counts"]:
            summary["asset_counts"][k] += cnts.get(k, 0)

        # write window pkl
        dst_split_logs = dst_logs_root / split
        safe_mkdir(dst_split_logs)
        dst_pkl = dst_split_logs / f"{anchor_token}.pkl"
        if not dst_pkl.exists():
            with open(dst_pkl, "wb") as f:
                pickle.dump(win_frames_copy, f, protocol=pickle.HIGHEST_PROTOCOL)

        # write meta
        dst_split_meta = dst_meta_root / split
        safe_mkdir(dst_split_meta)
        meta = {
            "category": category,
            "split": split,
            "scene_name": scene_name,
            "anchor_token": anchor_token,
            "window_start_frame_index_in_scene": start_idx,
            "window_end_frame_index_in_scene": end_idx,
            "num_history_frames": args.num_history_frames,
            "num_future_frames": args.num_future_frames,
        }
        (dst_split_meta / f"{anchor_token}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        tok_by_cat[category][split].add(anchor_token)
        summary["built"] += 1

    # write tokens_by_category
    for cat, smap in tok_by_cat.items():
        for split, toks in smap.items():
            cat_dir = dst_tokcat_root / cat
            safe_mkdir(cat_dir)
            (cat_dir / f"{split}.txt").write_text("\n".join(sorted(toks)) + "\n", encoding="utf-8")

    (dst_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("navlt_win root:", dst_root)


if __name__ == "__main__":
    main()
