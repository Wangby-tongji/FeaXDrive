"""Utilities to build a local (ego-frame) drivable Signed Distance Field (SDF).

This module supports two equivalent sources for inference-time FeaXDrive guidance:
  1) MetricCache: use the PDMDrivableMap serialized in NAVSIM metric cache.
  2) map_api/data scene: rebuild the same local PDMDrivableMap from the scene map_api.

Both paths finally produce an ego-frame SDF grid intended to be sampled with
``torch.nn.functional.grid_sample`` inside the diffusion sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap


@dataclass
class DrivableSDFConfig:
    # ROI in ego frame (rear-axle at t0): x-forward, y-left.
    x_min: float = -10.0
    x_max: float = 80.0
    y_min: float = -30.0
    y_max: float = 30.0
    resolution: float = 0.2  # [m]
    # Which map layers define "drivable" for guidance.
    layers: Tuple[SemanticMapLayer, ...] = (
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA,
        SemanticMapLayer.CARPARK_AREA,
    )
    # Numerical safety: clamp SDF for stable gradients.
    sdf_clip: float = 50.0


def _iter_polygons(geom) -> Iterable[Polygon]:
    """Yield Polygon(s) from shapely Polygon / MultiPolygon / GeometryCollection-like."""
    if geom is None:
        return
    gtype = getattr(geom, "geom_type", "")
    if gtype == "Polygon":
        yield geom
    elif gtype == "MultiPolygon":
        for p in geom.geoms:
            yield p
    else:
        geoms = getattr(geom, "geoms", None)
        if geoms is not None:
            for gg in geoms:
                yield from _iter_polygons(gg)


def _poly_to_cv_pts(
    poly: Polygon,
    *,
    x_min: float,
    y_max: float,
    res: float,
    H: int,
    W: int,
) -> Optional[np.ndarray]:
    """Convert a shapely Polygon exterior to OpenCV pixel points [N,1,2] int32."""
    if poly.is_empty:
        return None
    coords = np.asarray(poly.exterior.coords, dtype=np.float64)
    if coords.shape[0] < 3:
        return None

    col = (coords[:, 0] - x_min) / res
    row = (y_max - coords[:, 1]) / res
    pts = np.stack([col, row], axis=1)

    pts = np.rint(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts.reshape(-1, 1, 2)


def build_local_drivable_sdf_context_from_drivable_map(
    drivable_area_map: PDMDrivableMap,
    ego_state: EgoState,
    device: torch.device,
    cfg: Optional[DrivableSDFConfig] = None,
) -> Dict[str, object]:
    """Build an ego-frame SDF grid from a PDMDrivableMap.

    Args:
        drivable_area_map: local/global shapely polygon container.
        ego_state: current ego state; its rear axle defines the ego-frame origin.
        device: target torch device.
        cfg: SDF rasterization parameters.

    Returns:
        Dict with keys ``valid``, ``sdf``, ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``res``, ``H``, ``W``.
    """
    cfg = cfg or DrivableSDFConfig()

    try:
        dm = drivable_area_map
        ego0 = ego_state.rear_axle

        res = float(cfg.resolution)
        x_min, x_max = float(cfg.x_min), float(cfg.x_max)
        y_min, y_max = float(cfg.y_min), float(cfg.y_max)
        W = int(np.ceil((x_max - x_min) / res)) + 1
        H = int(np.ceil((y_max - y_min) / res)) + 1

        mask = np.zeros((H, W), dtype=np.uint8)

        keep_idcs = dm.get_indices_of_map_type(list(cfg.layers))
        if len(keep_idcs) == 0:
            return {"valid": False}

        for idx in keep_idcs:
            token = dm.tokens[idx]
            geom = dm[token]
            if geom is None or getattr(geom, "is_empty", False):
                continue

            # Transform global map geometry into ego rear-axle frame.
            g_local = translate(geom, xoff=-float(ego0.x), yoff=-float(ego0.y))
            g_local = rotate(g_local, -float(ego0.heading), origin=(0.0, 0.0), use_radians=True)

            for poly in _iter_polygons(g_local):
                pts = _poly_to_cv_pts(poly, x_min=x_min, y_max=y_max, res=res, H=H, W=W)
                if pts is not None:
                    cv2.fillPoly(mask, [pts], color=1)

        din = distance_transform_edt(mask) * res
        dout = distance_transform_edt(1 - mask) * res
        sdf = np.clip(din - dout, -float(cfg.sdf_clip), float(cfg.sdf_clip)).astype(np.float32)
        sdf_t = torch.from_numpy(sdf)[None, None, ...].to(device=device, dtype=torch.float32)

        return {
            "valid": True,
            "sdf": sdf_t,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "res": res,
            "H": H,
            "W": W,
        }
    except Exception:
        return {"valid": False}


def build_local_drivable_sdf_context(
    metric_cache: MetricCache,
    device: torch.device,
    cfg: Optional[DrivableSDFConfig] = None,
) -> Dict[str, object]:
    """Build an ego-frame drivable SDF grid from a NAVSIM MetricCache.

    This is the original paper-reproduction path. The metric cache stores the local
    ``PDMDrivableMap`` used by the PDMS/DAC evaluator. Only the map polygons and
    current ego state are used for guidance; future ground-truth trajectory and
    score information are not used.
    """
    return build_local_drivable_sdf_context_from_drivable_map(
        metric_cache.drivable_area_map,
        metric_cache.ego_state,
        device=device,
        cfg=cfg,
    )


def build_local_drivable_sdf_context_from_map_api(
    map_api: AbstractMap,
    ego_state: EgoState,
    device: torch.device,
    cfg: Optional[DrivableSDFConfig] = None,
    map_radius: float = 100.0,
) -> Dict[str, object]:
    """Build an ego-frame drivable SDF grid directly from scene data/map_api.

    This avoids using the serialized ``MetricCache`` as the guidance source. It
    reconstructs a ``PDMDrivableMap`` from the same current-scene HD map and ego
    state, then rasterizes it into the SDF used by FeaXDrive guidance.
    """
    try:
        dm = PDMDrivableMap.from_simulation(map_api, ego_state, map_radius=float(map_radius))
        return build_local_drivable_sdf_context_from_drivable_map(dm, ego_state, device=device, cfg=cfg)
    except Exception:
        return {"valid": False}
