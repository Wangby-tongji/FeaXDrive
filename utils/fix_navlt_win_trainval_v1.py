#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, pickle
from pathlib import Path
from collections import defaultdict

META_DIR = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win/meta/test")
RAW_LOGS = Path("/path/to/ReCogDrive/navsim/dataset/navsim_logs/test")
OUT = Path("/path/to/ReCogDrive/navsim/dataset/navlt_win/token2rawscene_test.json")

def main():
    # 统计每个 raw scene(2021...) 里有哪些 token 需要查
    scene2tokens = defaultdict(set)
    for mf in META_DIR.glob("*.json"):
        token = mf.stem
        meta = json.loads(mf.read_text(encoding="utf-8"))
        scene_folder = meta["scene_name"]
        scene2tokens[scene_folder].add(token)

    print("unique scene folders:", len(scene2tokens))

    token2rawscene = {}
    missed = 0

    for scene_folder, tokens in scene2tokens.items():
        pkl = RAW_LOGS / f"{scene_folder}.pkl"
        if not pkl.exists():
            missed += len(tokens)
            continue

        frames = pickle.load(open(pkl, "rb"))
        # 建一个 token->scene_name 的索引（只关心我们需要的 tokens）
        need = set(tokens)
        for fr in frames:
            tk = fr.get("token")
            if tk in need:
                token2rawscene[tk] = fr.get("scene_name")
                need.remove(tk)
                if not need:
                    break
        missed += len(need)

    OUT.write_text(json.dumps(token2rawscene), encoding="utf-8")
    print("mapped tokens:", len(token2rawscene))
    print("missed tokens:", missed)
    print("saved:", OUT)

if __name__ == "__main__":
    main()
