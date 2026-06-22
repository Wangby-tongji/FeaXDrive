#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_IMPROMPTU = Path("/path/to/ReCogDrive/navsim/dataset/impromptu")
DEFAULT_OUT = Path("/path/to/ReCogDrive/navsim/dataset/navlt")

# 从 original_image_paths.txt 里解析 NAVSIM 的 log folder
# 例：.../navsim/sensor_blobs/<split>/<log_folder>/CAM_F0/<xxx>.jpg
LOG_RE = re.compile(r"/navsim/sensor_blobs/[^/]+/([^/]+)/CAM_F0/")

def sanitize_name(s: str) -> str:
    """清理文件夹名：把空格/斜杠等替换掉，避免路径问题"""
    s = s.strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[\/\\\:\*\?\"\<\>\|]+", "_", s)  # Windows/Unix 都安全
    s = re.sub(r"__+", "_", s)
    return s or "unknown"

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore").strip()

def parse_navsim_f0_path(original_paths_file: Path, sid: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从 <sid>.original_image_paths.txt 里找 CAM_F0 对应的 NAVSIM jpg 路径，并解析 log_folder
    返回: (navsim_jpg_path, navsim_log_folder)
    """
    if not original_paths_file.exists():
        return None, None

    navsim_path = None
    for line in read_text(original_paths_file).splitlines():
        # 行格式示例：
        # NAVSIM_0_11373.CAM_F0.png: /.../navsim/sensor_blobs/.../<log>/CAM_F0/xxxx.jpg
        if ".CAM_F0.png:" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                navsim_path = parts[1].strip()
            break

    log_folder = None
    if navsim_path:
        m = LOG_RE.search(navsim_path)
        if m:
            log_folder = m.group(1)

    return navsim_path, log_folder

def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        # 默认用软链接，节省空间
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass
        except OSError:
            # 如果目标文件系统不支持软链接，退化为复制
            shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impromptu_root", type=Path, default=DEFAULT_IMPROMPTU)
    ap.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                    help="symlink: 软链接（默认，省空间）；copy: 复制文件")
    args = ap.parse_args()

    impromptu_root: Path = args.impromptu_root
    out_root: Path = args.out_root
    mode: str = args.mode

    if not impromptu_root.exists():
        raise FileNotFoundError(f"Impromptu root not found: {impromptu_root}")

    # 输出目录
    scenarios_root = out_root / "scenarios"       # 结构化：类别/场景/图片
    category_all_root = out_root / "category_all" # 平铺：类别/全部图片

    scenarios_root.mkdir(parents=True, exist_ok=True)
    category_all_root.mkdir(parents=True, exist_ok=True)

    # 记录：每个类别的汇总 csv writer（后面一次性写也行；这里用缓存列表更简单）
    category_rows = {}  # cat -> list[tuple(sample_id, imp_png, navsim_jpg, scenario_dir_rel)]

    # 扫描所有样本：以 *.category.txt 为基准
    cat_files = sorted(impromptu_root.rglob("*.category.txt"))
    print(f"[INFO] Found {len(cat_files)} category files under {impromptu_root}")

    n_ok, n_skip_no_f0, n_skip_no_cat = 0, 0, 0

    for cat_file in cat_files:
        shard_dir = cat_file.parent
        fname = cat_file.name
        # sid = 去掉 ".category.txt"
        if not fname.endswith(".category.txt"):
            continue
        sid = fname[:-len(".category.txt")]  # e.g., NAVSIM_0_11373

        category = read_text(cat_file)
        if not category:
            n_skip_no_cat += 1
            continue

        # 读取 clip_id（如果没有也不致命）
        clip_file = shard_dir / f"{sid}.clip_id.txt"
        clip_id = read_text(clip_file) if clip_file.exists() else "unknown_clip"

        # F0 png
        f0_png = shard_dir / f"{sid}.CAM_F0.png"
        if not f0_png.exists():
            n_skip_no_f0 += 1
            continue

        # NAVSIM 路径映射
        orig_paths_file = shard_dir / f"{sid}.original_image_paths.txt"
        navsim_jpg, log_folder = parse_navsim_f0_path(orig_paths_file, sid)
        log_folder = log_folder or "unknown_log"

        # 组装 scenario key：category + clip + log_folder
        cat_name = sanitize_name(category)
        clip_name = sanitize_name(clip_id)
        log_name = sanitize_name(log_folder)

        scenario_dir = scenarios_root / cat_name / f"{log_name}__{clip_name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # 1) 把该样本 F0 放进 scenario 文件夹（命名用 sid，保证唯一）
        out_png_in_scenario = scenario_dir / f"{sid}.png"
        link_or_copy(f0_png, out_png_in_scenario, mode=mode)

        # 2) 同时放进 category_all/类别/ （平铺汇总）
        cat_all_dir = category_all_root / cat_name
        cat_all_dir.mkdir(parents=True, exist_ok=True)
        out_png_in_cat_all = cat_all_dir / f"{sid}.png"
        link_or_copy(f0_png, out_png_in_cat_all, mode=mode)

        # 记录行：scenario 里也要 csv；category_all 也要 csv
        rel_scenario = scenario_dir.relative_to(out_root)
        row = (sid, str(out_png_in_cat_all), navsim_jpg or "", str(rel_scenario))
        category_rows.setdefault(cat_name, []).append(row)

        # scenario 自己的 csv（追加写）
        scenario_csv = scenario_dir / "navsim_paths.csv"
        write_header = not scenario_csv.exists()
        with scenario_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["sample_id", "impromptu_f0_png", "navsim_f0_jpg_path"])
            w.writerow([sid, str(out_png_in_scenario), navsim_jpg or ""])

        n_ok += 1

    # 写每个类别的汇总 CSV
    for cat_name, rows in category_rows.items():
        cat_dir = category_all_root / cat_name
        csv_path = cat_dir / "navsim_paths.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "category_all_f0_png", "navsim_f0_jpg_path", "scenario_dir_rel_to_navlt"])
            for r in rows:
                w.writerow(r)

    print("[DONE]")
    print(f"  OK samples: {n_ok}")
    print(f"  Skipped (no CAM_F0.png): {n_skip_no_f0}")
    print(f"  Skipped (empty category): {n_skip_no_cat}")
    print(f"  Output root: {out_root}")
    print(f"    - scenarios:     {scenarios_root}")
    print(f"    - category_all:  {category_all_root}")


if __name__ == "__main__":
    main()
