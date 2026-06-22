from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import pickle
import io
import time
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.agents.feaxdrive.utils.drivable_sdf import (
    DrivableSDFConfig,
    build_local_drivable_sdf_context,
    build_local_drivable_sdf_context_from_map_api,
)
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.evaluate.dynamics_metrics_paper import compute_dynamics_metrics

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _sync_cuda(device=None):
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device=device)
        except Exception:
            torch.cuda.synchronize()


def _tic(device=None) -> float:
    _sync_cuda(device)
    return time.perf_counter()


def _toc_ms(start: float, device=None) -> float:
    _sync_cuda(device)
    return (time.perf_counter() - start) * 1000.0

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    tokens_to_evaluate = sorted(tokens_to_evaluate)

    pdm_results: List[Dict[str, Any]] = []
    timing_results: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens_to_evaluate):
        if dist.get_rank() == 0:
            logger.info(f"Rank {dist.get_rank()} processing scenario {idx+1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}")

        timer_device = torch.device(agent.device) if torch.cuda.is_available() else None
        timing_row: Dict[str, Any] = {
            "token": token,
            "valid": True,
            "rank": dist.get_rank(),
            "cache_hidden_state": bool(getattr(agent, "cache_hidden_state", False)),
        }

        planner_cfg = getattr(getattr(agent, "action_head", None), "config", None)
        if planner_cfg is not None:
            timing_row.update({
                "sampling_method": getattr(planner_cfg, "sampling_method", None),
                "pred_type": getattr(planner_cfg, "pred_type", None),
                "num_inference_steps": int(getattr(planner_cfg, "num_inference_steps", 0)),
                "use_drivable_guidance": bool(getattr(planner_cfg, "use_drivable_guidance", False)),
                "drivable_guidance_last_k": int(getattr(planner_cfg, "drivable_guidance_last_k", 0)),
            })

        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            token_t0 = _tic(timer_device)

            cache_t0 = _tic(timer_device)
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            timing_row["timing_metric_cache_load_ms"] = _toc_ms(cache_t0, timer_device)

            timing_row["timing_drivable_sdf_build_ms"] = 0.0
            try:
                if hasattr(agent, "set_drivable_sdf_context") and hasattr(agent, "drivable_sdf_cfg"):
                    use_g = bool(
                        getattr(getattr(agent, "action_head", None), "config", None)
                        and getattr(agent.action_head.config, "use_drivable_guidance", False)
                    )
                    if use_g:
                        sdf_t0 = _tic(timer_device)
                        sdf_cfg = DrivableSDFConfig(**agent.drivable_sdf_cfg)
                        sdf_source = str(getattr(agent, "drivable_sdf_source", "metric_cache")).lower()
                        timing_row["drivable_sdf_source"] = sdf_source

                        if sdf_source in {"scene_map", "data_map", "data", "scene"}:
                            # Rebuild the local drivable SDF from NAVSIM scene data + nuPlan map_api.
                            # This avoids using the serialized evaluator MetricCache as the guidance source.
                            scene_for_sdf = scene_loader.get_scene_from_token(token)
                            scenario_for_sdf = NavSimScenario(
                                scene_for_sdf,
                                map_root=os.environ["NUPLAN_MAPS_ROOT"],
                                map_version=os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"),
                            )
                            ctx = build_local_drivable_sdf_context_from_map_api(
                                scenario_for_sdf.map_api,
                                scenario_for_sdf.get_ego_state_at_iteration(0),
                                device=torch.device(agent.device),
                                cfg=sdf_cfg,
                                map_radius=float(getattr(agent, "drivable_sdf_map_radius", 100.0)),
                            )
                        else:
                            # Paper-reproduction default: use the PDMDrivableMap stored in MetricCache.
                            ctx = build_local_drivable_sdf_context(
                                metric_cache,
                                device=torch.device(agent.device),
                                cfg=sdf_cfg,
                            )

                        timing_row["timing_drivable_sdf_build_ms"] = _toc_ms(sdf_t0, timer_device)
                        agent.set_drivable_sdf_context(ctx)
                    else:
                        agent.set_drivable_sdf_context(None)
            except Exception as e:
                timing_row["timing_drivable_sdf_build_ms"] = np.nan
                timing_row["drivable_sdf_error"] = repr(e)
                logger.warning(f"Failed to build drivable SDF for token {token}: {e}")

            requires_scene = False
            agent_input = scene_loader.get_agent_input_from_token(token)
            agent_t0 = _tic(timer_device)
            if requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)
            timing_row["timing_agent_total_ms_external"] = _toc_ms(agent_t0, timer_device)

            if hasattr(agent, "last_timing_stats") and agent.last_timing_stats is not None:
                timing_row.update(agent.last_timing_stats)

            if hasattr(agent, "last_delta_pi_mean") and agent.last_delta_pi_mean is not None:
                score_row["delta_pi_mean"] = float(agent.last_delta_pi_mean)
            if hasattr(agent, "last_delta_pi_curve") and agent.last_delta_pi_curve is not None:
                score_row["delta_pi_curve"] = agent.last_delta_pi_curve

            if hasattr(cfg, "dynamics_eval") and getattr(cfg.dynamics_eval, "enable", False):
                dt = float(getattr(cfg.dynamics_eval, "dt", trajectory.trajectory_sampling.interval_length))
                v_max = getattr(cfg.dynamics_eval, "v_max", None)
                a_max = getattr(cfg.dynamics_eval, "a_max", None)
                j_max = getattr(cfg.dynamics_eval, "j_max", None)
                kappa_max = getattr(cfg.dynamics_eval, "kappa_max", None)
                kappa_geo_max = getattr(cfg.dynamics_eval, "kappa_geo_max", None)
                kappa_adapt = bool(getattr(cfg.dynamics_eval, "kappa_adapt", False))
                a_lat_max = getattr(cfg.dynamics_eval, "a_lat_max", None)
                smooth_ksize = int(getattr(cfg.dynamics_eval, "smooth_ksize", 5))
                dyn = compute_dynamics_metrics(
                    poses=trajectory.poses,
                    dt=dt,
                    v_max=v_max,
                    a_max=a_max,
                    j_max=j_max,
                    kappa_max=kappa_max,
                    kappa_geo_max=kappa_geo_max,
                    kappa_adapt=kappa_adapt,
                    a_lat_max=a_lat_max,
                    smooth_ksize=smooth_ksize,
                )
                score_row.update({f"dyn_{k}": v for k, v in dyn.items()})

            pdm_t0 = _tic(timer_device)
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            timing_row["timing_pdm_score_ms"] = _toc_ms(pdm_t0, timer_device)
            timing_row["timing_eval_token_total_ms"] = _toc_ms(token_t0, timer_device)

            score_row.update(asdict(pdm_result))
            if hasattr(agent, "last_delta_pi_curve") and getattr(agent, "last_delta_pi_curve") is not None:
                score_row["delta_pi_curve"] = getattr(agent, "last_delta_pi_curve")
            if hasattr(agent, "last_delta_pi_mean") and getattr(agent, "last_delta_pi_mean") is not None:
                score_row["delta_pi_mean"] = getattr(agent, "last_delta_pi_mean")
            score_row["rank"] = dist.get_rank()
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False
            timing_row["valid"] = False

        pdm_results.append(score_row)
        timing_results.append(timing_row)

    payload = {
        "scores": pdm_results,
        "timings": timing_results,
    }
    return pickle.dumps(payload)


def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    """
    Helper function to broadcast an object from the source rank to all other processes.
    :param obj: Object to broadcast.
    :param device: Device to use for tensor operations.
    :param src: Source rank.
    :return: Broadcasted object.
    """
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor)).to(device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0).to(device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(size_tensor.item()).to(device)
        dist.broadcast(tensor, src=src)
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    return obj


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    build_logger(cfg)
    # worker = build_worker(cfg)


    scene_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=instantiate(cfg.train_test_split.scene_filter),
            sensor_config=SensorConfig.build_no_sensors(),
        )
    if rank == 0:
        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        tokens_to_evaluate = sorted(tokens_to_evaluate)  
        num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if num_missing_metric_cache_tokens > 0:
            logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
        if num_unused_metric_cache_tokens > 0:
            logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)


    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))

    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token] 
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token],
        })

    serialized_payload = run_pdm_score(data_points)

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    serialized_tensor = torch.ByteTensor(list(serialized_payload)).to(device)

    local_size = len(serialized_tensor)
    local_size_tensor = torch.tensor([local_size], dtype=torch.long, device=device)
    size_list = [torch.zeros_like(local_size_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size_tensor)
    size_list = [int(x.item()) for x in size_list]

    max_size = max(size_list)

    if local_size < max_size:
        padded_tensor = torch.cat([serialized_tensor, torch.zeros(max_size - local_size, dtype=torch.uint8, device=device)])
    else:
        padded_tensor = serialized_tensor

    gathered_results = [torch.empty_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, padded_tensor)

    if dist.get_rank() == 0:
        final_results = []
        final_timings = []
        for gathered_tensor, valid_size in zip(gathered_results, size_list):
            serialized_data = gathered_tensor[:valid_size].cpu().numpy().tobytes()
            payload = pickle.loads(serialized_data)
            final_results.extend(payload.get("scores", []))
            final_timings.extend(payload.get("timings", []))

        pdm_score_df = pd.DataFrame(final_results)
        timing_df = pd.DataFrame(final_timings)

        num_sucessful_scenarios = pdm_score_df["valid"].sum()
        num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
        df_avg = pdm_score_df.drop(columns=["token", "valid", "rank"], errors="ignore")

        numeric_cols = df_avg.select_dtypes(include=["number"]).columns
        average_row = df_avg[numeric_cols].mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = pdm_score_df["valid"].all()
        average_row["rank"] = "0"
        pdm_score_df.loc[len(pdm_score_df)] = average_row

        save_path = Path(cfg.output_dir)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        score_csv_path = save_path / f"{timestamp}.csv"
        timing_csv_path = save_path / f"{timestamp}_timing.csv"
        pdm_score_df.to_csv(score_csv_path, index=False)

        if len(timing_df) > 0:
            timing_avg_base = timing_df.drop(
                columns=[
                    "token", "valid", "rank",
                    "sampling_method", "pred_type",
                    "use_drivable_guidance",
                    "cache_hidden_state",
                ],
                errors="ignore",
            )
            timing_numeric_cols = timing_avg_base.select_dtypes(include=["number"]).columns
            timing_average_row = timing_avg_base[timing_numeric_cols].mean(skipna=True)
            timing_average_row["token"] = "average"
            timing_average_row["valid"] = timing_df["valid"].all() if "valid" in timing_df.columns else True
            timing_average_row["rank"] = "all"

            if "sampling_method" in timing_df.columns and len(timing_df) > 0:
                timing_average_row["sampling_method"] = timing_df["sampling_method"].iloc[0]
            if "pred_type" in timing_df.columns and len(timing_df) > 0:
                timing_average_row["pred_type"] = timing_df["pred_type"].iloc[0]
            if "use_drivable_guidance" in timing_df.columns and len(timing_df) > 0:
                timing_average_row["use_drivable_guidance"] = timing_df["use_drivable_guidance"].iloc[0]
            if "cache_hidden_state" in timing_df.columns and len(timing_df) > 0:
                timing_average_row["cache_hidden_state"] = timing_df["cache_hidden_state"].iloc[0]
            if "drivable_sdf_source" in timing_df.columns and len(timing_df) > 0:
                timing_average_row["drivable_sdf_source"] = timing_df["drivable_sdf_source"].dropna().iloc[0] if len(timing_df["drivable_sdf_source"].dropna()) > 0 else ""

            timing_df.loc[len(timing_df)] = timing_average_row
            timing_df.to_csv(timing_csv_path, index=False)

        if "delta_pi_curve" in pdm_score_df.columns:
            curves = []
            for _, row in pdm_score_df.iterrows():
                if row.get("token") == "average":
                    continue
                c = row.get("delta_pi_curve")
                if isinstance(c, list) and len(c) > 0:
                    curves.append(c)
            if len(curves) > 0:
                L = max(len(c) for c in curves)
                arr = np.zeros((len(curves), L), dtype=np.float64)
                mask = np.zeros((len(curves), L), dtype=np.float64)
                for i, c in enumerate(curves):
                    l = len(c)
                    arr[i, :l] = np.asarray(c, dtype=np.float64)
                    mask[i, :l] = 1.0
                mean_curve = (arr.sum(axis=0) / np.maximum(mask.sum(axis=0), 1.0)).tolist()
                pd.DataFrame({"step": list(range(L)), "delta_pi_mean": mean_curve}).to_csv(
                    save_path / f"{timestamp}_delta_pi_curve.csv", index=False
                )

        logger.info(
            f"""
            Finished running evaluation.
                Number of successful scenarios: {num_sucessful_scenarios}.
                Number of failed scenarios: {num_failed_scenarios}.
                Final average score of valid results: {pdm_score_df['score'].mean()}.
                Score CSV: {score_csv_path}.
                Timing CSV: {timing_csv_path if len(timing_df) > 0 else 'not generated'}.
            """
        )



if __name__ == "__main__":
    main()
