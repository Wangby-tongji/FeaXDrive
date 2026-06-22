#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# -------- utils --------
def safe_category_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "UNKNOWN"
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", " ", s)
    return s

def parse_impromptu_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse one path line from *.original_image_paths.txt.
    Returns (split, scene_name, cam_name, filename)
    Supports:
      test/<scene>/CAM_F0/<file>.jpg
    and absolute paths containing .../sensor_blobs/test/<scene>/CAM_F0/<file>.jpg
    """
    s = (line or "").strip()
    if not s:
        return None
    parts = s.split("/")

    # find split position
    idx = None
    split = None
    for i, p in enumerate(parts):
        if p in ("train", "trainval", "test"):
            idx = i
            split = p
            break
    if idx is None or split is None:
        return None
    if idx + 2 >= len(parts):
        return None

    scene_name = parts[idx + 1]
    cam_name = parts[idx + 2]
    filename = parts[-1]

    # map train -> trainval
    if split == "train":
        split = "trainval"
    return split, scene_name, cam_name, filename

def choose_key_line(lines: List[str]) -> Optional[str]:
    """Prefer CAM_F0 line; else first non-empty."""
    cand = []
    for ln in lines:
        if ln.strip():
            cand.append(ln.strip())
    if not cand:
        return None
    for ln in cand:
        if "/CAM_F0/" in ln or ln.endswith(".jpg") and "CAM_F0" in ln:
            return ln
    return cand[0]

def find_frame_token_by_image(frames: List[dict], scene_name: str, cam_name: str, filename: str) -> Optional[str]:
    """
    Find the frame whose cams[cam_name]['data_path'] ends with:
      <scene_name>/<cam_name>/<filename>
    Return frames[i]['token'].
    """
    target_suffix = f"{scene_name}/{cam_name}/{filename}"
    for fr in frames:
        cams = fr.get("cams", {})
        cam = cams.get(cam_name, {})
        dp = cam.get("data_path", "")
        if dp and dp.endswith(target_suffix):
            return fr.get("token")
    return None

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impromptu_root", type=str, required=True,
                    help="Extracted impromptu root (contains navsim_train_shard_xxxx folders)")
    ap.add_argument("--navsim_logs_test", type=str, required=True)
    ap.add_argument("--navsim_logs_trainval", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output root where per-category token lists & mapping jsonl will be written.")
    args = ap.parse_args()

    imp_root = Path(args.impromptu_root).expanduser().resolve()
    logs_root = {
        "test": Path(args.navsim_logs_test).expanduser().resolve(),
        "trainval": Path(args.navsim_logs_trainval).expanduser().resolve(),
    }

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # category -> split -> sets
    cat_tokens: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    cat_scenes: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    # write mapping
    out_jsonl = out_root / "lt_mapping.jsonl"
    missing_pkl = 0
    missing_match = 0
    total_samples = 0

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for cat_file in imp_root.rglob("*.category.txt"):
            total_samples += 1
            prefix = cat_file.name[: -len(".category.txt")]
            sample_dir = cat_file.parent

            try:
                category = cat_file.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                category = cat_file.read_text(encoding="latin-1").strip()
            category = safe_category_name(category)

            paths_file = sample_dir / f"{prefix}.original_image_paths.txt"
            if not paths_file.exists():
                continue

            try:
                lines = paths_file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                lines = paths_file.read_text(encoding="latin-1").splitlines()

            key_line = choose_key_line(lines)
            if not key_line:
                continue

            parsed = parse_impromptu_line(key_line)
            if not parsed:
                continue
            split, scene_name, cam_name, filename = parsed

            pkl_path = logs_root[split] / f"{scene_name}.pkl"
            if not pkl_path.exists():
                missing_pkl += 1
                record = {
                    "category": category, "split": split,
                    "scene_name": scene_name, "cam": cam_name, "filename": filename,
                    "status": "missing_pkl", "pkl_path": str(pkl_path)
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            frames = pickle.load(open(pkl_path, "rb"))
            if not isinstance(frames, list) or not frames:
                missing_match += 1
                record = {
                    "category": category, "split": split,
                    "scene_name": scene_name, "cam": cam_name, "filename": filename,
                    "status": "bad_pkl_format"
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            tok = find_frame_token_by_image(frames, scene_name, cam_name, filename)
            if not tok:
                missing_match += 1
                record = {
                    "category": category, "split": split,
                    "scene_name": scene_name, "cam": cam_name, "filename": filename,
                    "status": "no_frame_match"
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # aggregate
            cat_tokens[category][split].add(tok)
            cat_scenes[category][split].add(scene_name)

            record = {
                "category": category,
                "split": split,
                "scene_name": scene_name,
                "anchor_token": tok,
                "cam": cam_name,
                "filename": filename,
                "status": "ok",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # write per-category lists
    for category, split_map in cat_tokens.items():
        for split, toks in split_map.items():
            cat_dir = out_root / category / split
            cat_dir.mkdir(parents=True, exist_ok=True)

            (cat_dir / "tokens.txt").write_text("\n".join(sorted(toks)) + "\n", encoding="utf-8")
            (cat_dir / "scene_names.txt").write_text("\n".join(sorted(cat_scenes[category][split])) + "\n", encoding="utf-8")

    summary = {
        "total_samples_scanned": total_samples,
        "missing_pkl": missing_pkl,
        "missing_frame_match": missing_match,
        "out_jsonl": str(out_jsonl),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
