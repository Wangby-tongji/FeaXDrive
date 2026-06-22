#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import pickle
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

# 解析 impromptu original_image_paths 里的 CAM_F0 jpg 路径：
# .../sensor_blobs/{trainval|test}/{log}/CAM_F0/{token}.jpg
F0_RE = re.compile(r"/sensor_blobs/(trainval|test)/([^/]+)/CAM_F0/([0-9a-fA-F]+)\.jpg$")

CAM_KEYS = ["CAM_F0","CAM_L0","CAM_L1","CAM_L2","CAM_R0","CAM_R1","CAM_R2","CAM_B0"]

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip()

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)

def find_navsim_log_pkl(navsim_logs_split_dir: Path, log_name: str) -> Optional[Path]:
    # 常见：navsim_logs/<split>/<log>.pkl
    cand1 = navsim_logs_split_dir / f"{log_name}.pkl"
    if cand1.exists():
        return cand1
    # 兼容：navsim_logs/<split>/<log>/log.pkl
    cand2 = navsim_logs_split_dir / log_name / "log.pkl"
    if cand2.exists():
        return cand2
    return None

def parse_impromptu_one(prefix: Path):
    """
    prefix: .../NAVSIM_x_yyyyy (无后缀)
    读取：
      - .original_image_paths.txt （只用 CAM_F0 那条）
      - .category.txt
    """
    orig_p = Path(str(prefix) + ".original_image_paths.txt")
    cat_p  = Path(str(prefix) + ".category.txt")
    if not orig_p.exists() or not cat_p.exists():
        return None

    category = read_text(cat_p)
    f0_path = None
    for line in read_text(orig_p).splitlines():
        if ".CAM_F0.png:" in line:
            f0_path = line.split(":", 1)[1].strip()
            break
    if not f0_path:
        return None

    m = F0_RE.search(f0_path)
    if not m:
        return {"ok": False, "reason": "F0 path not match pattern", "impromptu_id": prefix.name, "category": category, "f0_path": f0_path}

    split, log_name, f0_token = m.group(1), m.group(2), m.group(3)
    return {"ok": True, "impromptu_id": prefix.name, "category": category, "split": split, "log_name": log_name, "f0_token": f0_token, "f0_path": f0_path}

def stem_token_from_path(rel_path: str) -> str:
    # rel_path like: "<log>/CAM_F0/<token>.jpg"
    return Path(rel_path).stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impromptu_root", type=Path, required=True)
    ap.add_argument("--navsim_root", type=Path, required=True)
    ap.add_argument("--out_name", type=str, default="navlt_imp_pure")
    ap.add_argument("--history_frames", type=int, default=4,
                    help="为每个长尾t0样本额外收集多少历史帧(含t0)。默认4，对齐ReCogDrive历史4帧。设为1则只收集t0。")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    imp_root = args.impromptu_root
    nav_root = args.navsim_root

    navsim_logs = nav_root / "navsim_logs"
    sensor_blobs = nav_root / "sensor_blobs"

    out_root = nav_root / args.out_name
    out_logs = out_root / "navsim_logs"
    out_blobs = out_root / "sensor_blobs"
    out_meta = out_root / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)

    # 1) 扫 impromptu
    orig_files = list(imp_root.rglob("*.original_image_paths.txt"))
    if not orig_files:
        raise RuntimeError(f"No *.original_image_paths.txt found under {imp_root}")

    imp_records, bad_records = [], []
    for of in orig_files:
        prefix = Path(str(of).replace(".original_image_paths.txt", ""))
        rec = parse_impromptu_one(prefix)
        if rec is None:
            continue
        (imp_records if rec.get("ok") else bad_records).append(rec)

    # 2) 按 split/log 分组，避免重复读pkl
    by_log = defaultdict(list)
    cat_cnt = defaultdict(int)
    for r in imp_records:
        by_log[(r["split"], r["log_name"])].append(r)
        cat_cnt[r["category"]] += 1

    # 3) 从 pkl 定位：F0_token -> frame_index -> ego_token，并收集8cam+lidar路径
    # 输出 index：以 ego_token 为主键（SceneFilter.tokens 用它）
    index_rows = []
    need_paths_by_split = defaultdict(set)  # split -> set(rel_path)  (相机+lidar)
    need_logs_by_split = defaultdict(set)
    ego_tokens_by_split = defaultdict(set)

    # 用于统计/检查
    miss_f0_in_pkl = 0
    miss_file_on_disk = 0

    for (split, log_name), recs in by_log.items():
        pkl = find_navsim_log_pkl(navsim_logs / split, log_name)
        if pkl is None:
            print(f"[WARN] log pkl not found: {split} {log_name}")
            continue

        scene_dict_list = pickle.load(open(pkl, "rb"))  # list[dict]

        # 建 CAM_F0 token -> frame idx 映射（从 pkl 的 cams['CAM_F0']['data_path']）
        f0tok2idx = {}
        for i, fr in enumerate(scene_dict_list):
            cams = fr.get("cams", {})
            if "CAM_F0" in cams and "data_path" in cams["CAM_F0"]:
                tok = stem_token_from_path(cams["CAM_F0"]["data_path"])
                f0tok2idx[tok] = i

        for r in recs:
            f0_token = r["f0_token"]
            if f0_token not in f0tok2idx:
                miss_f0_in_pkl += 1
                continue

            t0_idx = f0tok2idx[f0_token]
            ego_token = scene_dict_list[t0_idx]["token"]  # 这是 SceneFilter 用的 token

            # 收集历史帧索引（含t0）
            H = max(1, args.history_frames)
            start = t0_idx - (H - 1)
            if start < 0:
                start = 0
            frame_indices = list(range(start, t0_idx + 1))

            # 收集每个历史帧的 8cam + lidar
            for fi in frame_indices:
                fr = scene_dict_list[fi]
                cams = fr["cams"]
                # 8 cameras
                for ck in CAM_KEYS:
                    if ck not in cams or "data_path" not in cams[ck]:
                        continue
                    rel = cams[ck]["data_path"]  # 相对于 sensor_blobs/<split> 的路径，通常含 <log>/CAM_X/<token>.jpg
                    need_paths_by_split[split].add(rel)

                # merged lidar point cloud
                rel_lidar = fr.get("lidar_path", None)
                if rel_lidar:
                    need_paths_by_split[split].add(rel_lidar)

            # 写索引行（保留category/impromptu_id等）
            index_rows.append({
                "split": split,
                "log_name": log_name,
                "impromptu_id": r["impromptu_id"],
                "category": r["category"],
                "f0_token": f0_token,
                "ego_token": ego_token,
                "t0_frame_index_in_log": t0_idx,
            })

            ego_tokens_by_split[split].add(ego_token)
            need_logs_by_split[split].add(log_name)

    # 4) 写 meta
    stats = {
        "num_impromptu_records": len(imp_records),
        "num_index_rows": len(index_rows),
        "num_bad_impromptu_records": len(bad_records),
        "miss_f0_token_in_pkl": miss_f0_in_pkl,
        "num_trainval_ego_tokens": len(ego_tokens_by_split["trainval"]),
        "num_test_ego_tokens": len(ego_tokens_by_split["test"]),
        "num_trainval_logs": len(need_logs_by_split["trainval"]),
        "num_test_logs": len(need_logs_by_split["test"]),
        "category_counts": dict(sorted(cat_cnt.items(), key=lambda x: -x[1])),
        "history_frames_included": args.history_frames,
        "note": "ego_token用于SceneFilter.tokens；8相机与lidar_path均来自pkl的data_path/lidar_path，避免相机token不一致问题",
    }
    (out_meta / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # index.csv
    with (out_meta / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split","log_name","impromptu_id","category","f0_token","ego_token","t0_frame_index_in_log"])
        w.writeheader()
        for row in index_rows:
            w.writerow({k: row.get(k,"") for k in w.fieldnames})

    # tokens list (for SceneFilter.tokens)
    for split in ["trainval", "test"]:
        (out_meta / f"tokens_{split}.txt").write_text("\n".join(sorted(ego_tokens_by_split[split])), encoding="utf-8")
        (out_meta / f"logs_{split}.txt").write_text("\n".join(sorted(need_logs_by_split[split])), encoding="utf-8")

    if bad_records:
        (out_meta / "bad_impromptu_records.json").write_text(json.dumps(bad_records, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print("[DRY RUN] no filesystem changes")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    # 5) 组织 navlt_imp_pure：链接需要的 pkl + 需要的 sensor 文件
    # 5.1 link navsim_logs pkl（按log）
    for split in ["trainval", "test"]:
        src_split = navsim_logs / split
        dst_split = out_logs / split
        dst_split.mkdir(parents=True, exist_ok=True)

        for log_name in sorted(need_logs_by_split[split]):
            pkl = find_navsim_log_pkl(src_split, log_name)
            if pkl is None:
                print(f"[WARN] pkl not found: {split} {log_name}")
                continue
            dst = dst_split / f"{log_name}.pkl"
            safe_symlink(pkl, dst)

    # 5.2 link sensor files（按相对路径 rel_path）
    # rel_path 本身通常包含 <log>/CAM_*/xxx.jpg 或 <log>/MergedPointCloud/xxx.pcd
    for split in ["trainval", "test"]:
        for rel in sorted(need_paths_by_split[split]):
            src = sensor_blobs / split / rel
            dst = out_blobs / split / rel
            if not src.exists():
                miss_file_on_disk += 1
                if miss_file_on_disk <= 30:
                    print(f"[WARN] missing file: {src}")
                continue
            safe_symlink(src, dst)

    print("[OK] built:", out_root)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if miss_file_on_disk:
        print(f"[WARN] missing sensor files on disk = {miss_file_on_disk}")
    if miss_f0_in_pkl:
        print(f"[WARN] f0 tokens not found in pkl = {miss_f0_in_pkl} (see stats)")

if __name__ == "__main__":
    main()
