"""Dynamics/comfort/feasibility metrics for trajectory validation.

These metrics are designed to validate the effectiveness of dynamics constraints
and projection-based diffusion sampling.

The core outputs are:
  - speed/acceleration/jerk norms (optional)
  - curvature and lateral acceleration (recommended for long-tail)
  - violation rates and average excess over limits

All computations are based on (x, y) positions in local coordinates.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch / numpy / list to numpy array (CPU).

    The PDM scoring script may pass torch.Tensor poses (from GPU) directly.
    These helpers make the dynamics metrics robust to that.
    """
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _binomial_kernel(ksize: int) -> np.ndarray:
    """Binomial (Pascal) smoothing kernel."""
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError(f"smooth_ksize must be odd and >=3, got {ksize}")
    n = ksize - 1
    # use integer binomial coefficients; stable for small ksize
    from math import comb
    coeff = np.array([comb(n, i) for i in range(ksize)], dtype=np.float64)
    coeff = coeff / np.sum(coeff)
    return coeff


def _smooth_1d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Replicate-padding 1D smoothing along time axis."""
    pad = len(kernel) // 2
    if pad <= 0:
        return arr
    left = np.repeat(arr[:1], pad, axis=0)
    right = np.repeat(arr[-1:], pad, axis=0)
    padded = np.concatenate([left, arr, right], axis=0)
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(arr.shape[0]):
        window = padded[i:i + len(kernel)]
        out[i] = np.sum(window * kernel[:, None], axis=0)
    return out


def compute_curvature_alat_arrays(
    poses: Any,
    dt: float,
    smooth_ksize: int = 5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute curvature kappa and lateral acceleration a_lat (scheme B).

    Uses: smooth -> arc length -> variable-step finite differences.

    Args:
        poses: [T,3] (x,y,heading)
        dt: seconds per step
        smooth_ksize: odd smoothing kernel size
    Returns:
        kappa: [T]
        a_lat: [T]
    """
    poses = _to_numpy(poses)
    xy = poses[:, :2].astype(np.float64)
    T = xy.shape[0]
    if T < 3:
        return np.zeros((T,), dtype=np.float64), np.zeros((T,), dtype=np.float64)

    k = _binomial_kernel(smooth_ksize)
    xy_s = _smooth_1d(xy, k)

    ds = np.linalg.norm(xy_s[1:] - xy_s[:-1], axis=-1)
    ds = np.maximum(ds, eps)
    s = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(ds)], axis=0)  # [T]

    # central diff for first derivative in s
    denom1 = np.maximum(s[2:] - s[:-2], eps)  # [T-2]
    p_prime = (xy_s[2:] - xy_s[:-2]) / denom1[:, None]  # [T-2,2]

    # variable-step second derivative
    ds_f = np.maximum(s[2:] - s[1:-1], eps)
    ds_b = np.maximum(s[1:-1] - s[:-2], eps)
    term_f = (xy_s[2:] - xy_s[1:-1]) / ds_f[:, None]
    term_b = (xy_s[1:-1] - xy_s[:-2]) / ds_b[:, None]
    p_dbl = 2.0 * (term_f - term_b) / denom1[:, None]

    x1, y1 = p_prime[:, 0], p_prime[:, 1]
    x2, y2 = p_dbl[:, 0], p_dbl[:, 1]
    num = x1 * y2 - y1 * x2
    denom = np.power(np.maximum(x1 * x1 + y1 * y1, eps), 1.5)
    k_mid = num / denom  # [T-2]

    # speed from arc length (geometric)
    v_seg = ds / dt  # [T-1]
    v_mid = 0.5 * (v_seg[:-1] + v_seg[1:])  # [T-2]
    a_lat_mid = (v_mid ** 2) * k_mid

    kappa = np.zeros((T,), dtype=np.float64)
    a_lat = np.zeros((T,), dtype=np.float64)
    kappa[1:-1] = k_mid
    a_lat[1:-1] = a_lat_mid
    return kappa, a_lat


def _finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    return (arr[1:] - arr[:-1]) / dt


def compute_dynamics_arrays(
    poses: Any,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute v/a/j arrays from poses.

    Args:
        poses: [T, 3] (x, y, heading)
        dt: seconds
    Returns:
        v: [T-1,2], a: [T-2,2], j: [T-3,2]
    """
    poses = _to_numpy(poses)
    xy = poses[:, :2].astype(np.float64)
    v = _finite_diff(xy, dt)
    a = _finite_diff(v, dt) if len(v) >= 2 else np.zeros((0, 2), dtype=np.float64)
    j = _finite_diff(a, dt) if len(a) >= 2 else np.zeros((0, 2), dtype=np.float64)
    return v, a, j


def compute_dynamics_metrics(
    poses: Any,
    dt: float,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
    j_max: Optional[float] = None,
    kappa_max: Optional[float] = None,
    kappa_geo_max: Optional[float] = None,
    kappa_adapt: bool = False,
    a_lat_max: Optional[float] = None,
    smooth_ksize: int = 5,
) -> Dict[str, Any]:
    """Compute scalar metrics for a single trajectory.

    User-facing convention (as requested):
      - report *max* magnitude (not mean/p95)
      - treat a trajectory as violating a limit if *any* time step exceeds it
        (binary 0/1), without reporting excess amplitude.
    """
    v, a, j = compute_dynamics_arrays(poses, dt)

    v_norm = np.linalg.norm(v, axis=-1) if v.size else np.array([], dtype=np.float64)
    a_norm = np.linalg.norm(a, axis=-1) if a.size else np.array([], dtype=np.float64)
    j_norm = np.linalg.norm(j, axis=-1) if j.size else np.array([], dtype=np.float64)

    def _stats(norm: np.ndarray, max_val: Optional[float], name: str) -> Dict[str, Any]:
        """Return max value and any-step violation indicator."""
        if norm.size == 0:
            return {
                f"{name}_max": 0.0,
                f"{name}_any_violation": 0.0,
            }
        max_mag = float(np.max(norm))
        if max_val is None:
            any_v = 0.0
        else:
            any_v = float(np.any(norm > float(max_val)))
        return {
            f"{name}_max": max_mag,
            f"{name}_any_violation": any_v,
        }

    out: Dict[str, Any] = {}
    out.update(_stats(v_norm, v_max, "v"))
    out.update(_stats(a_norm, a_max, "a"))
    out.update(_stats(j_norm, j_max, "j"))

    # ---- Curvature / lateral acceleration (scheme B: arc-length + smooth derivatives) ----
    # These are often the primary violations even when v/a/j are within bounds.
    kappa, a_lat = compute_curvature_alat_arrays(poses, dt=dt, smooth_ksize=smooth_ksize)
    k_abs = np.abs(kappa)
    a_abs = np.abs(a_lat)
    out.update(_stats(k_abs, kappa_max, "kappa"))
    out.update(_stats(a_abs, a_lat_max, "a_lat"))

    # ---- Speed-adaptive curvature bound (more physical):
    #   kappa_max_adapt(t) = min(kappa_geo_max, a_lat_max / (v(t)^2 + eps))
    # and we report a *rate* of violating time steps.
    if bool(kappa_adapt) and (kappa_geo_max is not None or a_lat_max is not None):
        poses_np = _to_numpy(poses)
        xy = poses_np[:, :2].astype(np.float64)
        T = xy.shape[0]
        if T >= 3:
            # Reuse the same scheme-B arc-length speed estimate as compute_curvature_alat_arrays.
            k = _binomial_kernel(smooth_ksize)
            xy_s = _smooth_1d(xy, k)
            ds = np.linalg.norm(xy_s[1:] - xy_s[:-1], axis=-1)
            ds = np.maximum(ds, 1e-6)
            v_seg = ds / float(dt)  # [T-1]
            v_mid = 0.5 * (v_seg[:-1] + v_seg[1:])  # [T-2]
            v = np.zeros((T,), dtype=np.float64)
            v[1:-1] = v_mid
            # Build kappa_max_adapt(t)
            bounds = []
            if kappa_geo_max is not None:
                bounds.append(np.full((T,), float(kappa_geo_max), dtype=np.float64))
            if a_lat_max is not None:
                bounds.append(float(a_lat_max) / (v * v + 1e-6))
            kappa_bound = bounds[0]
            for b in bounds[1:]:
                kappa_bound = np.minimum(kappa_bound, b)
            # Violation rate over time steps.
            out["kappa_adapt_violation_rate"] = float(np.mean(np.abs(kappa) > kappa_bound))
        else:
            out["kappa_adapt_violation_rate"] = 0.0
    return out
