#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import List, Optional

import hydra
from hydra.utils import instantiate
from hydra import initialize_config_dir

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def init_distributed() -> (int, int, int):
    """torchrun 下自动初始化；普通 python 运行则 world_size=1"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"

        dist.init_process_group(backend=backend, init_method="env://")
        return rank, world_size, local_rank

    return 0, 1, 0


def load_scene_filter(filter_name: str, config_dir: Path) -> SceneFilter:
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name=filter_name)
    return instantiate(cfg)


def read_tokens(tokens: Optional[List[str]],
                tokens_file: Optional[str],
                tokens_csv: Optional[str],
                token_col: str) -> List[str]:
    out: List[str] = []

    if tokens:
        out.extend([t.strip() for t in tokens if str(t).strip()])

    if tokens_file:
        p = Path(tokens_file)
        text = p.read_text(encoding="utf-8", errors="ignore")
        # 支持：一行一个 / 空格分隔 / 逗号分隔
        for raw in text.replace(",", " ").split():
            s = raw.strip()
            if s:
                out.append(s)

    if tokens_csv:
        import pandas as pd
        df = pd.read_csv(tokens_csv)
        if token_col not in df.columns:
            raise KeyError(f"token_col '{token_col}' not found in {tokens_csv}. columns={list(df.columns)}")
        out.extend([str(x).strip() for x in df[token_col].dropna().tolist()])

    # 去重且保持顺序
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def main():
    parser = argparse.ArgumentParser()
    # 数据与筛选
    parser.add_argument("--data_root", type=str, default=None,
                        help="数据根目录（包含 sensor_blobs/ 与 navsim_logs/）。默认读环境变量 OPENSCENE_DATA_ROOT")
    parser.add_argument("--split", type=str, default="test", choices=["mini", "test", "trainval"])
    parser.add_argument("--filter", type=str, default="all_scenes",
                        help="scene_filter 配置名（不带 .yaml）")
    parser.add_argument("--filter_config_dir", type=str, default=None,
                        help="scene_filter 配置目录。默认: $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/train_test_split/scene_filter")

    # agent 配置（都从 bash 传）
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--vlm_path", type=str, required=True)
    parser.add_argument("--cam_type", type=str, default="single")
    parser.add_argument("--vlm_type", type=str, default="internvl")
    parser.add_argument("--dit_type", type=str, default="small")
    parser.add_argument("--vlm_size", type=str, default="small")
    parser.add_argument("--sampling_method", type=str, default="ddim")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--cache_hidden_state", action="store_true")

    # 可视化输入：token 支持三种方式
    parser.add_argument("--tokens", nargs="*", default=None, help="直接指定一个或多个 ego token")
    parser.add_argument("--tokens_file", type=str, default=None, help="txt 文件：一行一个 token（或空格/逗号分隔）")
    parser.add_argument("--tokens_csv", type=str, default=None, help="csv 文件：从某列读取 token")
    parser.add_argument("--token_col", type=str, default="ego_token", help="tokens_csv 的 token 列名")

    # 输出
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dpi", type=int, default=200)

    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # data_root
    data_root = args.data_root or os.getenv("OPENSCENE_DATA_ROOT", "")
    if not data_root:
        raise ValueError("data_root is empty. Please set --data_root or export OPENSCENE_DATA_ROOT")
    data_root = Path(data_root)

    # filter config dir
    if args.filter_config_dir:
        filter_dir = Path(args.filter_config_dir)
    else:
        navsim_root = os.getenv("NAVSIM_DEVKIT_ROOT", "")
        if not navsim_root:
            raise ValueError("NAVSIM_DEVKIT_ROOT not set. Please export it or pass --filter_config_dir")
        filter_dir = Path(navsim_root) / "navsim/planning/script/config/common/train_test_split/scene_filter"

    scene_filter = load_scene_filter(args.filter, filter_dir)

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
        data_root / f"navsim_logs/{args.split}",
        data_root / f"sensor_blobs/{args.split}",
        scene_filter,
        sensor_config=sensor_cfg,
    )
    scene_loader_traj = SceneLoader(
        data_root / f"navsim_logs/{args.split}",
        data_root / f"sensor_blobs/{args.split}",
        scene_filter,
        sensor_config=sensor_cfg,
        load_image_path=True,
    )

    tokens = read_tokens(args.tokens, args.tokens_file, args.tokens_csv, args.token_col)
    if len(tokens) == 0:
        raise ValueError("No tokens provided. Use --tokens / --tokens_file / --tokens_csv")

    # 分布式切分 token（与你原来的 rank::world_size 一致）
    local_tokens = tokens[rank::world_size]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 可视化
    import traceback

    with torch.no_grad():
        # 可选：提前拿到可用 token 集合（更快定位 split/filter 问题）
        loader_tokens = set(getattr(scene_loader, "tokens", []))

        for token in local_tokens:
            if loader_tokens and token not in loader_tokens:
                print(
                    f"[Rank {rank}] skip token={token} (not in scene_loader.tokens). "
                    f"Check split/filter/data_root."
                )
                continue

            # 1) load scene
            try:
                scene = scene_loader.get_scene_from_token(token)
            except Exception as e:
                print(f"[Rank {rank}] skip token={token} scene_load_failed: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)   # 你想更详细可改成 limit=None
                continue

            # 2) load scene_traj
            try:
                scene_traj = scene_loader_traj.get_scene_from_token(token)
            except Exception as e:
                print(f"[Rank {rank}] skip token={token} scene_traj_load_failed: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
                continue

            # 3) visualization
            try:
                frame_idx = scene.scene_metadata.num_history_frames - 1
                fig, _, _ = plot_bev_and_camera_with_agent(scene, scene_traj, frame_idx, agent)
            except Exception as e:
                print(f"[Rank {rank}] skip token={token} plot_failed: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
                continue

            save_path = out_dir / f"{token}_vis_traj.png"
            plt.savefig(str(save_path), bbox_inches="tight", dpi=args.dpi)
            plt.close(fig)
            print(f"[Rank {rank}] saved: {save_path}")

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()