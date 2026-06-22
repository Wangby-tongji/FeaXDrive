
# -*- coding: utf-8 -*-
"""
Curvature (kappa) via arc-length method (dpsi/ds) + lateral acceleration + speed metrics.
Python 3.9 compatible. Accepts torch.Tensor or numpy arrays.

Reported columns:
- dyn_v_p99, dyn_v_max
- dyn_kappa_p99, dyn_kappa_max
- dyn_a_lat_p99, dyn_a_lat_max
- dyn_kappa_violation_rate, dyn_a_lat_violation_rate

Violation_rate is defined as ANY_VIOLATION:
  single traj -> 1.0 if any effective timestep violates threshold
  batch -> fraction of samples with any violation

Notes on robustness:
- Curvature is computed as dpsi/ds with angle unwrapping.
- Steps with very small arc-length are ignored via ds_min to avoid numerical blow-ups.
- Quantiles/max are computed over effective steps only (NaNs ignored).
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import torch
except Exception:
    torch = None


def _to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _nanquantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.nanquantile(x, q))


def compute_speed_segments(xy: np.ndarray, dt: float) -> np.ndarray:
    """Speed on segments (T-1)."""
    if xy.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    d = xy[1:] - xy[:-1]
    v = np.linalg.norm(d, axis=1) / max(dt, 1e-9)
    return v.astype(np.float64)


def compute_kappa_arclength(xy: np.ndarray, dt: float, ds_min: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute curvature kappa (1/m) at midpoints using arc-length method:
        kappa ≈ dpsi/ds, where psi is heading of segment, s is arc-length.

    Returns:
      kappa: [T-2] signed curvature at midpoints (NaN for ineffective steps)
      v_mid: [T-2] speed at midpoints (m/s)
    """
    T = xy.shape[0]
    if T < 3:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    # Segment deltas and arc-lengths
    d = xy[1:] - xy[:-1]                       # [T-1,2]
    ds = np.linalg.norm(d, axis=1).astype(np.float64)  # [T-1]
    # Heading per segment
    psi = np.arctan2(d[:, 1], d[:, 0]).astype(np.float64)  # [T-1]
    psi = np.unwrap(psi)  # unwrap for stable dpsi

    # Midpoint speed (aligned with kappa indices 0..T-3)
    v_seg = ds / max(dt, 1e-9)
    v_mid = 0.5 * (v_seg[:-1] + v_seg[1:])  # [T-2]

    # Central-ish difference for kappa at midpoints:
    dpsi = psi[1:] - psi[:-1]  # [T-2]
    ds_mid = 0.5 * (ds[:-1] + ds[1:])  # [T-2]

    # Avoid tiny ds to prevent numerical blow-ups
    effective = ds_mid > float(ds_min)
    kappa = np.full((T - 2,), np.nan, dtype=np.float64)
    kappa[effective] = dpsi[effective] / np.maximum(ds_mid[effective], 1e-9)

    return kappa, v_mid.astype(np.float64)


def compute_dynamics_metrics(
    poses: Any,
    dt: float,
    kappa_max: Optional[float] = None,
    a_lat_max: Optional[float] = None,
    ds_min: float = 0.05,
    **_ignored_kwargs: Any,
) -> Dict[str, float]:
    """
    poses: [T,D] or [B,T,D], uses first two dims as (x,y) in meters.
    dt: seconds between steps.

    Thresholds:
      - kappa_max in 1/m  (kinematic limit)
      - a_lat_max in m/s^2 (dynamic/comfort limit)

    ds_min:
      - minimal arc-length at midpoint to consider curvature valid (meters).
        Helps suppress false huge kappas when the vehicle is nearly stationary.

    Extra kwargs (e.g., v_max/a_max/j_max) are accepted and ignored for compatibility.
    """
    arr = _to_numpy(poses)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"poses must be [T,D] or [B,T,D], got shape {arr.shape}")

    B, T, D = arr.shape

    v_p99_list = []
    v_max_list = []
    kappa_p99_list = []
    kappa_max_list = []
    alat_p99_list = []
    alat_max_list = []
    kappa_any_list = []
    alat_any_list = []

    for i in range(B):
        xy = arr[i, :, :2].astype(np.float64)

        v_seg = compute_speed_segments(xy, dt)
        v_p99_list.append(_nanquantile(v_seg, 0.99))
        v_max_list.append(float(np.nanmax(v_seg)) if v_seg.size else 0.0)

        kappa, v_mid = compute_kappa_arclength(xy, dt, ds_min=float(ds_min))
        kappa_abs = np.abs(kappa)  # NaNs propagate; nanquantile/nanmax will ignore
        a_lat = (v_mid ** 2) * kappa_abs  # NaNs propagate

        kappa_p99_list.append(_nanquantile(kappa_abs[~np.isnan(kappa_abs)], 0.99) if np.any(~np.isnan(kappa_abs)) else float('nan'))
        kappa_max_list.append(float(np.nanmax(kappa_abs)) if np.any(~np.isnan(kappa_abs)) else 0.0)
        alat_p99_list.append(_nanquantile(a_lat[~np.isnan(a_lat)], 0.99) if np.any(~np.isnan(a_lat)) else float('nan'))
        alat_max_list.append(float(np.nanmax(a_lat)) if np.any(~np.isnan(a_lat)) else 0.0)

        # ANY_VIOLATION over effective steps only
        if kappa_max is None:
            kappa_any = 0.0
        else:
            kappa_any = 1.0 if np.any(kappa_abs > float(kappa_max)) else 0.0

        if a_lat_max is None:
            alat_any = 0.0
        else:
            alat_any = 1.0 if np.any(a_lat > float(a_lat_max)) else 0.0

        kappa_any_list.append(kappa_any)
        alat_any_list.append(alat_any)

    out: Dict[str, float] = {
        "dyn_v_p99": float(np.nanmean(v_p99_list)),
        "dyn_v_max": float(np.nanmean(v_max_list)),
        "dyn_kappa_p99": float(np.nanmean(kappa_p99_list)),
        "dyn_kappa_max": float(np.nanmean(kappa_max_list)),
        "dyn_a_lat_p99": float(np.nanmean(alat_p99_list)),
        "dyn_a_lat_max": float(np.nanmean(alat_max_list)),
        "dyn_kappa_violation_rate": float(np.mean(kappa_any_list)),
        "dyn_a_lat_violation_rate": float(np.mean(alat_any_list)),
    }
    return out
