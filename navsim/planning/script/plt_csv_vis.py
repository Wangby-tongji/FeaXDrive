#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra import initialize_config_dir

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            dist.init_process_group(backend="gloo", init_method="env://")
        print("Distributed init: rank {}/{}, local_rank {}".format(rank, world_size, local_rank))
        return rank, world_size, local_rank
    return 0, 1, 0


def read_tokens(tokens, tokens_file, tokens_csv, token_col, max_tokens):
    out = []

    if tokens:
        for t in tokens:
            s = str(t).strip()
            if s:
                out.append(s)

    if tokens_file:
        p = Path(tokens_file)
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for raw in txt.replace(",", " ").split():
            s = raw.strip()
            if s:
                out.append(s)

    if tokens_csv:
        df = pd.read_csv(tokens_csv)
        if token_col not in df.columns:
            raise KeyError("token_col '{}' not in {}. columns={}".format(
                token_col, tokens_csv, list(df.columns)
            ))
        out.extend([str(x).strip() for x in df[token_col].dropna().tolist()])

    # 去重保持顺序
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    if max_tokens is not None and max_tokens > 0:
        uniq = uniq[:max_tokens]

    return uniq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["mini", "test", "trainval"])
    parser.add_argument("--filter", type=str, default="all_scenes", help="scene_filter 配置名（不带 .yaml）")

    # agent
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--vlm_path", type=str, required=True)
    parser.add_argument("--cam_type", type=str, default="single")
    parser.add_argument("--vlm_type", type=str, default="internvl")
    parser.add_argument("--dit_type", type=str, default="small")
    parser.add_argument("--vlm_size", type=str, default="small")
    parser.add_argument("--sampling_method", type=str, default="ddim")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--cache_hidden_state", action="store_true")

    # tokens input
    parser.add_argument("--tokens", nargs="*", default=None, help="直接传多个 token")
    parser.add_argument("--tokens_file", type=str, default=None, help="txt：一行一个 token（或空格/逗号分隔）")
    parser.add_argument("--tokens_csv", type=str, default=None, help="csv：从某列读取 token")
    parser.add_argument("--token_col", type=str, default="ego_token", help="tokens_csv 的列名")
    parser.add_argument("--max_tokens", type=int, default=None, help="只取前 K 个 token（可选）")

    # output
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--skip_existing", action="store_true", help="已有图片则跳过")

    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    openscene_data_root = os.getenv("OPENSCENE_DATA_ROOT", "")
    if not openscene_data_root:
        raise ValueError("OPENSCENE_DATA_ROOT not set")
    openscene_data_root = Path(openscene_data_root)

    # scene_filter config dir（沿用你原脚本结构）
    config_dir = Path(os.getenv("NAVSIM_DEVKIT_ROOT", "")) / \
                 "navsim/planning/script/config/common/train_test_split/scene_filter"
    if not config_dir.exists():
        raise FileNotFoundError("scene_filter config_dir not found: {}".format(config_dir))

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name=args.filter)
    scene_filter = instantiate(cfg)  # type: SceneFilter

    # agent
    agent = ReCogDriveAgent(
        TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path=args.checkpoint_path,
        vlm_path=args.vlm_path,
        cam_type='single',
        vlm_type='internvl',
        dit_type='small',
        sampling_method='ddim',
        diffusion_pred_type='x0',          # ✅ 必须对齐训练
        cache_hidden_state=True,           # ✅ bool
        cache_mode=True,                   # ✅ bool（在线算 hidden state）
        vlm_size='small',
        grpo=False,
    ).to(device)
    
    agent.eval()
    agent.initialize()

    sensor_cfg = agent.module.get_sensor_config() if hasattr(agent, "module") else agent.get_sensor_config()

    scene_loader = SceneLoader(
        openscene_data_root / "navsim_logs" / args.split,
        openscene_data_root / "sensor_blobs" / args.split,
        scene_filter,
        sensor_config=sensor_cfg,
    )
    scene_loader_traj = SceneLoader(
        openscene_data_root / "navsim_logs" / args.split,
        openscene_data_root / "sensor_blobs" / args.split,
        scene_filter,
        sensor_config=sensor_cfg,
        load_image_path=True,
    )

    tokens = read_tokens(args.tokens, args.tokens_file, args.tokens_csv, args.token_col, args.max_tokens)
    if len(tokens) == 0:
        raise ValueError("No tokens provided. Use --tokens / --tokens_file / --tokens_csv")

    # 过滤到当前 scene_loader 里能找到的 token（避免浪费时间）
    try:
        available = set(scene_loader.tokens)
        before = len(tokens)
        tokens = [t for t in tokens if t in available]
        if rank == 0:
            print("[INFO] tokens filtered by scene_loader.tokens: {} -> {}".format(before, len(tokens)))
    except Exception:
        pass

    local_tokens = tokens[rank::world_size]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for token in tqdm(local_tokens, desc="Rank {} processing tokens".format(rank)):
        save_path = out_dir / "{}_vis_traj.png".format(token)
        if args.skip_existing and save_path.exists():
            continue

        try:
            scene = scene_loader.get_scene_from_token(token)
            scene_traj = scene_loader_traj.get_scene_from_token(token)
        except Exception as e:
            print("[Rank {}] skip token={} (load failed): {}".format(rank, token, e))
            continue

        frame_idx = scene.scene_metadata.num_history_frames - 1
        with torch.no_grad():
            fig, _, _ = plot_bev_and_camera_with_agent(scene, scene_traj, frame_idx, agent)
        plt.savefig(str(save_path), bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()