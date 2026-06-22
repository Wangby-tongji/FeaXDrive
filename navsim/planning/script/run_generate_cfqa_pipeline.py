from __future__ import annotations

"""Standalone CFQA pipeline (v3.1): Unified EvidenceExtractor + 3 CFQA types.

What this script does
- Loads NAVSIM scenes directly via SceneLoader.get_scene_from_token.
- DOES NOT perform any 3D->camera projection (box_2d). This avoids failures when
  calibration fields (e.g., sensor2lidar_rotation) are missing in some samples.
- Extracts lightweight evidence using a unified EvidenceExtractor:
  1) object relevance to a corridor polyline (cone/ped/bicycle)
  2) visibility score (lightweight image heuristics)
  3) lane confidence score (lightweight image heuristics)
- Generates *three* counterfactual QAs per sample, covering:
  A) ambiguous road boundary (lane_conf)
  B) temporary rule/layout change (construction_score via cone relevance)
  C) adverse environment/road condition (visibility_score)

Key constraints
- Long-tail category is NOT provided as input.
- STOP is forbidden for low/mid risk. Must include STOP_COND always.
- Uses a corridor polyline provided by the caller. In this offline pipeline we
  use GT future trajectory polyline as a corridor proxy. For online inference,
  replace it with route/map centerline corridor.

Output format
- InternVL style conversations JSONL:
  {"id":..., "image": ["/abs/path.jpg"], "conversations": [{"from":"human","value":"<image>..."}, ...]}

Env
- Optional NAVLT_INDEX_CSV to restrict tokens to navlt_imp ego_token.
- Optional NAVLT_SPLIT to filter CSV by split.

"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
from omegaconf import DictConfig

import numpy as np

from navsim.common.dataloader import SceneLoader
from navsim.planning.utils.evidence_extractor import EvidenceExtractor

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


# -----------------
# Token selection
# -----------------

def _load_tokens_from_index_csv(index_csv: str, split: Optional[str] = None) -> List[str]:
    import pandas as pd
    df = pd.read_csv(index_csv)
    if "ego_token" not in df.columns:
        raise ValueError(f"index csv missing column ego_token: {index_csv}")
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == str(split)]
    return df["ego_token"].astype(str).dropna().unique().tolist()


# -----------------
# Policy helpers
# -----------------

def _stop_cond_default() -> str:
    return "仅当前方被完全阻挡或出现极近距离冲突时才允许STOP；否则必须持续推进。"


def _policy_from_risk(risk: str, nav_cmd: str) -> Dict[str, str]:
    """Map risk level to discrete action tags.

    nav_cmd constrains TURN actions: only allow TURN if nav_cmd matches.
    """
    if risk == "low":
        speed = "KEEP"  # light, progress-friendly
        progress = "COMMIT"
    elif risk == "mid":
        speed = "DECELERATE"
        progress = "CAUTIOUS_COMMIT"
    else:
        speed = "STOP"
        progress = "YIELD"

    # Path: by default STRAIGHT, only turn if commanded.
    if nav_cmd == "LEFT":
        path = "LEFT_TURN" if risk == "low" else "LEFT_TURN"
    elif nav_cmd == "RIGHT":
        path = "RIGHT_TURN" if risk == "low" else "RIGHT_TURN"
    else:
        path = "STRAIGHT"

    # Noise control: forbid STOP unless high
    if risk in ("low", "mid") and speed == "STOP":
        speed = "DECELERATE"
        progress = "CAUTIOUS_COMMIT"

    return {
        "SPEED": speed,
        "PATH": path,
        "PROGRESS": progress,
        "STOP_COND": _stop_cond_default(),
    }


def _render_world(world: str, key_feature: str, pol: Dict[str, str], why: str, extra: str = "") -> str:
    lines = [
        f"WORLD: {world}",
        f"KEY_FEATURE: {key_feature}",
        "POLICY:",
        f"  SPEED: {pol['SPEED']}",
        f"  PATH: {pol['PATH']}",
        f"  PROGRESS: {pol['PROGRESS']}",
        f"STOP_COND: {pol['STOP_COND']}",
        f"WHY: {why}",
    ]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _render_delta(from_pol: Dict[str, str], to_pol: Dict[str, str], diff: str) -> str:
    return "\n".join([
        "WORLD: DELTA",
        f"FROM: (SPEED={from_pol['SPEED']}, PATH={from_pol['PATH']}, PROGRESS={from_pol['PROGRESS']})",
        f"TO: (SPEED={to_pol['SPEED']}, PATH={to_pol['PATH']}, PROGRESS={to_pol['PROGRESS']})",
        f"DIFF: {diff}",
    ])


# -----------------
# CFQA generators
# -----------------

def _qa_boundary(ev: Dict[str, Any]) -> Tuple[str, str]:
    """Ambiguous boundary CFQA using lane_conf."""
    lane_conf = float(ev.get("lane_conf", 0.5))
    nav_cmd = str(ev.get("nav_cmd", "STRAIGHT"))

    # Counterfactual: boundary cues are clear
    cf_risk = "low"
    cf_pol = _policy_from_risk(cf_risk, nav_cmd)
    cf_key = "道路边界/车道线线索清晰（lane_conf高）"
    cf_why = "边界清晰时保持稳定推进即可，避免不必要降速导致进度下降。"

    # Factual: based on lane_conf
    if lane_conf < 0.2:
        risk = "mid"  # still forbid STOP by default; boundary alone should not force STOP
        key = "道路边界线索很弱/模糊（lane_conf低）"
    elif lane_conf < 0.35:
        risk = "mid"
        key = "道路边界线索偏弱（lane_conf偏低）"
    else:
        risk = "low"
        key = "道路边界线索尚可（lane_conf中高）"

    fact_pol = _policy_from_risk(risk, nav_cmd)
    change = "YES" if (cf_pol["SPEED"] != fact_pol["SPEED"] or cf_pol["PROGRESS"] != fact_pol["PROGRESS"]) else "NO"

    harm = "".join([
        "IF_NO_CHANGE_HARM: 可能出现左右摇摆、贴边行驶、临近才纠正导致急刹，进而拉低ego_progress。"
    ])
    fact_why = "边界不确定时应轻度/中度降速并约束横向摆动，沿稳定通行走廊持续推进（不应无意义停住）。"

    delta = _render_delta(cf_pol, fact_pol, "根据边界不确定性，SPEED/PROGRESS从‘正常推进’调整为‘更可控但持续推进’。")

    q = (
        "<image>\n"
        "请基于图像与证据，进行反事实推理并输出结构化通行策略。\n"
        "聚焦‘道路边界是否清晰’对策略的影响。\n"
        "必须输出：COUNTERFACTUAL 与 FACTUAL 两个世界的策略，并给出不改变的危害与怎么改；最后给出DELTA差分。\n"
        f"EVIDENCE: lane_conf={lane_conf:.2f}, visibility={float(ev.get('visibility_score',0.5)):.2f}, nav_cmd={nav_cmd}, ego_speed_bin={ev.get('ego_speed_bin','mid')}\n"
        "输出字段必须包含 SPEED/PATH/PROGRESS/STOP_COND。"
    )

    a = "\n\n".join([
        _render_world("COUNTERFACTUAL", cf_key, cf_pol, cf_why),
        "\n".join([
            "WORLD: FACTUAL",
            f"KEY_FEATURE: {key}",
            f"CHANGE: {change}",
            f"WHY: {fact_why}",
            harm,
            "NEW_POLICY:",
            f"  SPEED: {fact_pol['SPEED']}",
            f"  PATH: {fact_pol['PATH']}",
            f"  PROGRESS: {fact_pol['PROGRESS']}",
            f"STOP_COND: {fact_pol['STOP_COND']}",
        ]),
        delta,
    ])
    return q, a


def _qa_temporary_change(ev: Dict[str, Any]) -> Tuple[str, str]:
    """Temporary rule/layout change CFQA using construction_score + cone_rel."""
    nav_cmd = str(ev.get("nav_cmd", "STRAIGHT"))
    cone_rel = str(ev.get("cone_rel", "none"))
    cone_path = ev.get("min_cone_path_dist")
    cone_obj = ev.get("min_cone_obj_dist")
    constr = float(ev.get("construction_score", 0.0))

    cf_pol = _policy_from_risk("low", nav_cmd)
    cf_key = "未观察到明确施工/临时控制影响通行走廊"
    cf_why = "没有明显施工影响时保持正常推进，避免无意义停滞。"

    # Factual risk: only if cone near/on path -> mid/high
    if cone_rel == "on_path" and (cone_obj is not None and float(cone_obj) <= 12.0):
        risk = "high"
        key = "施工控制物可能阻挡通行走廊（cone_rel=on_path且距离近）"
    elif cone_rel in ("on_path", "near_path"):
        risk = "mid"
        key = "存在施工控制物靠近通行走廊（cone_rel=near/on）"
    else:
        risk = "low"
        key = "施工控制物与通行走廊相关性低（cone_rel=off/none）"

    fact_pol = _policy_from_risk(risk, nav_cmd)
    change = "YES" if (cf_pol["SPEED"] != fact_pol["SPEED"] or cf_pol["PROGRESS"] != fact_pol["PROGRESS"]) else "NO"

    harm = "IF_NO_CHANGE_HARM: 可能贴近封闭区/控制物导致擦碰；临近才反应引发急刹急转；或在入口处过度停滞导致ego_progress低。"
    fact_why = "施工/临时控制影响走廊时应中度降速并沿走廊中心稳定通过；除非明确被阻挡，否则不应长期停住。"

    delta = _render_delta(cf_pol, fact_pol, "根据施工控制物与走廊相关性，调整速度与推进承诺，避免卡死同时保证安全。")

    q = (
        "<image>\n"
        "请基于图像与证据，进行反事实推理并输出结构化通行策略。\n"
        "聚焦‘临时施工/改道是否影响通行走廊’对策略的影响。\n"
        "必须输出：COUNTERFACTUAL 与 FACTUAL 两个世界的策略，并给出不改变的危害与怎么改；最后给出DELTA差分。\n"
        f"EVIDENCE: cone_rel={cone_rel}, min_cone_path_dist={cone_path}, min_cone_obj_dist={cone_obj}, construction_score={constr:.2f}, nav_cmd={nav_cmd}\n"
        "输出字段必须包含 SPEED/PATH/PROGRESS/STOP_COND。"
    )

    a = "\n\n".join([
        _render_world("COUNTERFACTUAL", cf_key, cf_pol, cf_why),
        "\n".join([
            "WORLD: FACTUAL",
            f"KEY_FEATURE: {key}",
            f"CHANGE: {change}",
            f"WHY: {fact_why}",
            harm,
            "NEW_POLICY:",
            f"  SPEED: {fact_pol['SPEED']}",
            f"  PATH: {fact_pol['PATH']}",
            f"  PROGRESS: {fact_pol['PROGRESS']}",
            f"STOP_COND: {fact_pol['STOP_COND']}",
        ]),
        delta,
    ])
    return q, a


def _qa_adverse(ev: Dict[str, Any]) -> Tuple[str, str]:
    """Adverse environment/road condition CFQA using visibility_score (light).

    Note: surface_risk is a weak proxy here; can be improved later.
    """
    nav_cmd = str(ev.get("nav_cmd", "STRAIGHT"))
    vis = float(ev.get("visibility_score", 0.5))
    lane_conf = float(ev.get("lane_conf", 0.5))

    cf_pol = _policy_from_risk("low", nav_cmd)
    cf_key = "能见度/感知质量良好（visibility高）"
    cf_why = "能见度好时保持正常推进，避免无意义降速造成进度损失。"

    if vis < 0.18 and lane_conf < 0.25:
        risk = "high"
        key = "能见度极差且边界线索弱（visibility很低）"
    elif vis < 0.35:
        risk = "mid"
        key = "能见度下降/画面质量变差（visibility偏低）"
    else:
        risk = "low"
        key = "能见度尚可（visibility中高）"

    fact_pol = _policy_from_risk(risk, nav_cmd)
    change = "YES" if (cf_pol["SPEED"] != fact_pol["SPEED"] or cf_pol["PROGRESS"] != fact_pol["PROGRESS"]) else "NO"

    harm = "IF_NO_CHANGE_HARM: 可能看不清远处风险导致反应过晚、急刹急转；或在不确定中反复试探，造成ego_progress下降。"
    fact_why = "能见度差时应降低速度提高可控性并扩大安全余度，但除非极端不可见/被阻挡，否则仍应谨慎推进。"

    delta = _render_delta(cf_pol, fact_pol, "根据能见度/感知质量变化，调整速度与推进承诺，做到安全前提下持续推进。")

    q = (
        "<image>\n"
        "请基于图像与证据，进行反事实推理并输出结构化通行策略。\n"
        "聚焦‘能见度/环境退化’对策略的影响。\n"
        "必须输出：COUNTERFACTUAL 与 FACTUAL 两个世界的策略，并给出不改变的危害与怎么改；最后给出DELTA差分。\n"
        f"EVIDENCE: visibility_score={vis:.2f}, lane_conf={lane_conf:.2f}, nav_cmd={nav_cmd}, ego_speed_bin={ev.get('ego_speed_bin','mid')}\n"
        "输出字段必须包含 SPEED/PATH/PROGRESS/STOP_COND。"
    )

    a = "\n\n".join([
        _render_world("COUNTERFACTUAL", cf_key, cf_pol, cf_why),
        "\n".join([
            "WORLD: FACTUAL",
            f"KEY_FEATURE: {key}",
            f"CHANGE: {change}",
            f"WHY: {fact_why}",
            harm,
            "NEW_POLICY:",
            f"  SPEED: {fact_pol['SPEED']}",
            f"  PATH: {fact_pol['PATH']}",
            f"  PROGRESS: {fact_pol['PROGRESS']}",
            f"STOP_COND: {fact_pol['STOP_COND']}",
        ]),
        delta,
    ])
    return q, a


# -----------------
# Main
# -----------------

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Optionally override tokens using index.csv
    index_csv = os.environ.get("NAVLT_INDEX_CSV", "")
    navlt_split = os.environ.get("NAVLT_SPLIT", None)
    override_tokens: Optional[List[str]] = None
    if index_csv:
        try:
            override_tokens = _load_tokens_from_index_csv(index_csv, navlt_split)
            logger.info(f"Loaded {len(override_tokens)} tokens from {index_csv} split={navlt_split}")
        except Exception as e:
            logger.warning(f"Failed to load NAVLT tokens from {index_csv}: {e}")

    # Build scene loader
    scene_filter = cfg.train_test_split.scene_filter
    if override_tokens is not None:
        scene_filter.tokens = override_tokens
        # ensure no truncation
        scene_filter.max_scenes = None

    scene_loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=scene_filter,
        load_image_path=True,
    )

    extractor = EvidenceExtractor(corridor_half_width_m=1.8, near_margin_m=1.2, lookahead_m=30.0)

    out_dir = Path(os.getcwd())
    out_path = out_dir / "navlt_imp_cfqa_v3.jsonl"

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        # Iterate tokens directly from SceneLoader to avoid any 3D->camera projection.
        # This is the root-cause fix for missing calibration (sensor2lidar_rotation=None).
        for token in scene_loader.tokens:
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene.get_agent_input()
            ego_statuses = agent_input.ego_statuses
            cameras = agent_input.cameras

            # t0 image (CAM_F0)
            img_path = str(cameras[-1].cam_f0.image)

            # corridor polyline (offline proxy): GT future trajectory in local coords (x,y,yaw)
            try:
                traj = scene.get_future_trajectory(num_trajectory_frames=scene.scene_metadata.num_future_frames)
                poses = np.asarray(traj.poses, dtype=np.float32)
                corridor = [(float(x), float(y)) for x, y in poses[:, :2]]
                if len(corridor) < 2:
                    corridor = [(0.0, 0.0), (10.0, 0.0)]
            except Exception:
                corridor = [(0.0, 0.0), (10.0, 0.0)]

            ego_vel = ego_statuses[-1].ego_velocity
            nav_cmd = ego_statuses[-1].driving_command

            # Objects from privileged annotations at t0 (no camera projection needed)
            frame_idx = scene.scene_metadata.num_history_frames - 1
            anns = scene.frames[frame_idx].annotations

            # NOTE: anns.names/anns.boxes can be numpy arrays. Never use `or []` on numpy arrays.
            names_raw = getattr(anns, "names", None)
            if names_raw is None:
                obj_names = []
            else:
                try:
                    obj_names = [str(n) for n in names_raw.tolist()]
                except Exception:
                    obj_names = [str(n) for n in list(names_raw)]

            boxes_raw = getattr(anns, "boxes", None)
            if boxes_raw is None:
                obj_boxes = np.zeros((0, 7), dtype=np.float32)
            else:
                obj_boxes = np.asarray(boxes_raw, dtype=np.float32)
                if obj_boxes.ndim == 1:
                    obj_boxes = obj_boxes.reshape(1, -1)

            ev = extractor.extract(
                image_path=img_path,
                ego_velocity_xy=ego_vel,
                nav_cmd_onehot=nav_cmd,
                obj_names=obj_names,
                obj_boxes_xyzlwhh=obj_boxes,
                corridor_polyline_xy=corridor,
            )

            # Generate 3 CFQAs
            q1, a1 = _qa_boundary(ev)
            q2, a2 = _qa_temporary_change(ev)
            q3, a3 = _qa_adverse(ev)

            sample = {
                "id": f"{token}",
                "image": [img_path],
                "conversations": [
                    {"from": "human", "value": q1},
                    {"from": "gpt", "value": a1},
                    {"from": "human", "value": q2},
                    {"from": "gpt", "value": a2},
                    {"from": "human", "value": q3},
                    {"from": "gpt", "value": a3},
                ],
                "evidence": ev,  # keep for debugging/filtering
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(f"Wrote {n_written} CFQA samples to: {out_path}")


if __name__ == "__main__":
    main()
