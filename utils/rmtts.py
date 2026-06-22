#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle
from pathlib import Path
from collections import defaultdict

NAVLT = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win")

SCENES_TXT = NAVLT / "utils" / "missing_scenes_from_bad_tokens.txt"

META_DIRS = {
    "trainval": NAVLT / "meta" / "trainval",
    "test": NAVLT / "meta" / "test",
}
LOGS_DIRS = {
    "trainval": NAVLT / "navsim_logs" / "trainval",
    "test": NAVLT / "navsim_logs" / "test",
}
BLOBS_DIRS = {
    "trainval": NAVLT / "sensor_blobs" / "trainval",
    "test": NAVLT / "sensor_blobs" / "test",
}

CAM = "CAM_F0"
EXPECTED_FRAMES = 14

OUT_DIR = NAVLT / "utils" / "missing_frames_report"
OUT_JSON = OUT_DIR / "missing_frames_by_scene.json"
OUT_SUMMARY = OUT_DIR / "summary.txt"

def read_scenes_list(path: Path):
    scenes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scenes.append(line)
    return scenes

def build_scene_to_tokens(split: str):
    """Return dict: scene_folder -> list[token] from meta json files."""
    scene2tokens = defaultdict(list)
    meta_dir = META_DIRS[split]
    for mf in meta_dir.glob("*.json"):
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        scene = meta.get("scene_name")
        token = mf.stem
        if scene:
            scene2tokens[scene].append(token)
    return scene2tokens

def load_window_frames(split: str, token: str):
    """Load token-level window pkl. Return list[dict] frames or None."""
    pkl_path = LOGS_DIRS[split] / f"{token}.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        if not isinstance(frames, list):
            return None
        return frames
    except Exception:
        return None

def get_cam_filenames(frames):
    """Return list of filenames for CAM_F0, length==len(frames)."""
    names = []
    for fr in frames:
        cams = fr.get("cams", {})
        caminfo = cams.get(CAM, {})
        rel = caminfo.get("data_path")
        if isinstance(rel, str) and rel:
            names.append(Path(rel).name)
        else:
            names.append(None)
    return names

def check_missing(split: str, token: str, filenames):
    """Check existence in navlt_win/sensor_blobs/{split}/{token}/CAM_F0/filename."""
    token_cam_dir = BLOBS_DIRS[split] / token / CAM
    missing = []
    for i, fn in enumerate(filenames):
        if fn is None:
            missing.append((i, None, "no_data_path"))
            continue
        p = token_cam_dir / fn
        if not p.exists():
            missing.append((i, fn, "missing_file"))
    return missing

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scenes = read_scenes_list(SCENES_TXT)
    print(f"scenes listed: {len(scenes)}")

    # Build scene->tokens maps for both splits
    scene2tokens = {
        "trainval": build_scene_to_tokens("trainval"),
        "test": build_scene_to_tokens("test"),
    }

    report = {}  # scene -> {split -> {token -> {...}}}
    total_tokens = 0
    total_tokens_with_missing = 0

    for scene in scenes:
        report[scene] = {}
        for split in ["trainval", "test"]:
            tokens = scene2tokens[split].get(scene, [])
            if not tokens:
                continue
            report[scene][split] = {}
            for token in sorted(tokens):
                frames = load_window_frames(split, token)
                if frames is None:
                    # token exists in meta but no pkl
                    report[scene][split][token] = {
                        "status": "missing_pkl",
                        "missing_frames": [],
                        "expected_frames": EXPECTED_FRAMES,
                        "actual_frames_in_pkl": None,
                    }
                    total_tokens += 1
                    total_tokens_with_missing += 1
                    continue

                filenames = get_cam_filenames(frames)
                missing = check_missing(split, token, filenames)

                total_tokens += 1
                if missing:
                    total_tokens_with_missing += 1

                report[scene][split][token] = {
                    "status": "ok",
                    "expected_frames": EXPECTED_FRAMES,
                    "actual_frames_in_pkl": len(frames),
                    "missing_count": len(missing),
                    "missing_frames": [
                        {"frame_index": i, "filename": fn, "reason": rsn}
                        for (i, fn, rsn) in missing
                    ],
                }

        # 同时写每个 scene 的人类可读 txt
        scene_txt = OUT_DIR / f"{scene}.txt"
        lines = []
        lines.append(f"SCENE: {scene}")
        if not report[scene]:
            lines.append("  (no tokens found in meta for trainval/test)")
        for split in ["trainval", "test"]:
            if split not in report[scene]:
                continue
            lines.append(f"\n[{split}] tokens: {len(report[scene][split])}")
            for token, info in report[scene][split].items():
                if info.get("status") == "missing_pkl":
                    lines.append(f"  - {token}: MISSING PKL")
                    continue
                mc = info.get("missing_count", 0)
                lines.append(f"  - {token}: missing {mc}/{info.get('actual_frames_in_pkl', '?')} frames in {CAM}")
                if mc:
                    for m in info["missing_frames"]:
                        lines.append(f"      * frame[{m['frame_index']}]: {m['filename']} ({m['reason']})")
        scene_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # write json
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary
    summary_lines = []
    summary_lines.append(f"scenes: {len(scenes)}")
    summary_lines.append(f"tokens checked (trainval+test): {total_tokens}")
    summary_lines.append(f"tokens with missing (missing pkl or missing frames): {total_tokens_with_missing}")
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("DONE")
    print("JSON:", OUT_JSON)
    print("Summary:", OUT_SUMMARY)
    print("Per-scene txt in:", OUT_DIR)

if __name__ == "__main__":
    main()
