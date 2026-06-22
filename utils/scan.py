#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, pickle
from pathlib import Path
from collections import defaultdict

NAVLT = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win")
META_DIR = NAVLT/"meta/trainval"

RAW_LOGS = Path("/path/to/ReCogDrive/navsim/dataset/navsim_logs/trainval")
RAW_BLOBS = Path("/path/to/ReCogDrive/navsim/dataset/sensor_blobs")
SRC_SPLITS = ["trainval", "test"]

CAM = "CAM_F0"
EXPECTED = 14

OUT_GOOD = NAVLT/"good_tokens_trainval_camf0_14.txt"
OUT_BAD  = NAVLT/"bad_tokens_trainval_camf0_lt14.txt"

def exists_any(scene_folder: str, cam: str, fname: str) -> bool:
    for sp in SRC_SPLITS:
        p = RAW_BLOBS/sp/scene_folder/cam/fname
        if p.exists():
            return True
    return False

def load_window(scene_folder: str, s: int, e: int):
    frames = pickle.load(open(RAW_LOGS/f"{scene_folder}.pkl","rb"))
    w1 = frames[s:e+1]
    if len(w1)==EXPECTED:
        return w1
    w2 = frames[s:e]
    if len(w2)==EXPECTED:
        return w2
    return None

def main():
    metas = sorted(META_DIR.glob("*.json"))
    print("meta:", len(metas))

    good = []
    bad = []

    # cache raw logs per scene to reduce IO (optional but helps)
    cache = {}

    for mf in metas:
        token = mf.stem
        meta = json.loads(mf.read_text(encoding="utf-8"))
        scene = meta["scene_name"]
        s = meta["window_start_frame_index_in_scene"]
        e = meta["window_end_frame_index_in_scene"]

        raw_pkl = RAW_LOGS/f"{scene}.pkl"
        if not raw_pkl.exists():
            bad.append(token)
            continue

        if scene not in cache:
            cache[scene] = pickle.load(open(raw_pkl,"rb"))
        frames = cache[scene]
        w1 = frames[s:e+1]
        window = w1 if len(w1)==EXPECTED else frames[s:e]
        if len(window)!=EXPECTED:
            bad.append(token)
            continue

        ok = 0
        for fr in window:
            rel = fr["cams"][CAM]["data_path"]
            fname = Path(rel).name
            if exists_any(scene, CAM, fname):
                ok += 1

        if ok == EXPECTED:
            good.append(token)
        else:
            bad.append(token)

    OUT_GOOD.write_text("\n".join(good) + "\n", encoding="utf-8")
    OUT_BAD.write_text("\n".join(bad) + "\n", encoding="utf-8")
    print("GOOD:", len(good), "BAD:", len(bad))
    print("saved:", OUT_GOOD)
    print("saved:", OUT_BAD)

if __name__ == "__main__":
    main()
