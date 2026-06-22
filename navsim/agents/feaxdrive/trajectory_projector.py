"""Unified differentiable trajectory projector for FeaXDrive.

This file intentionally merges the two historical projector implementations:

1) ``trajectory_projector_v4.py``
   - training-time dynamics projector;
   - curvature / lateral-acceleration violation regularization;
   - ``violation()`` used by ``lambda_dyn``;
   - ``project_dynamics()`` used by projection-consistency losses.

2) ``trajectory_projector_v4drive.py``
   - inference-time footprint-level drivable-area projector;
   - SDF context support;
   - ``project_drivable()`` used when an SDF context is attached.

The public ``project()`` method is a compatibility dispatcher. Without a drivable
context it behaves like v4; with a drivable context it behaves like v4drive.
Use explicit ``project_dynamics(...)`` or ``project_drivable(...)`` when a caller
must avoid ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import os
import math

import torch
import torch.nn.functional as F


def _is_rank0() -> bool:
    return os.getenv('RANK', '0') == '0' and os.getenv('LOCAL_RANK', '0') == '0'


def _dbg() -> bool:
    return os.getenv('DRIVABLE_SDF_DEBUG', '0') not in ('0', 'false', 'False', '')


@dataclass
class DrivableSDFContext:
    """Drivable context.

    Two supported modes:

    (A) Fixed-grid SDF context (what older code builds):
        - sdf: [B,1,H,W] tensor
        - x_min, y_min, resolution

    (B) Polygon context (enables dynamic SDF window build):
        - poly_local: shapely geometry in ego-local coordinates
        - resolution

    You may provide both; fixed-grid SDF takes precedence.
    """

    # Fixed-grid SDF (optional)
    sdf: Optional[torch.Tensor] = None
    x_min: float = 0.0
    y_min: float = 0.0

    # Polygon (optional)
    poly_local: Optional[Any] = None  # shapely geometry

    # Shared
    resolution: float = 0.25
    margin: float = 0.0


@dataclass
class ProjectionLimits:
    dt: float = 0.5
    # (kept for compatibility; not used by drivable projection)
    kappa_max: Optional[float] = None
    kappa_geo_max: Optional[float] = None
    a_lat_max: Optional[float] = None
    use_kappa_adapt: bool = False


@dataclass
class ProxConfig:
    steps: int = 1
    step_size: float = 0.1
    lambda_kappa: float = 1.0   # reused as drivable penalty weight
    lambda_a_lat: float = 1.0   # unused
    smooth_ksize: int = 5
    keep_start: bool = True
    eps: float = 1e-6


@dataclass
class _ActiveGrid:
    sdf: torch.Tensor      # [1,1,H,W]
    x_min: float
    y_min: float
    resolution: float
    H: int
    W: int


class TrajectoryProjector:
    """Differentiable proximal projector for drivable feasibility."""

    def __init__(self, limits: ProjectionLimits, prox: ProxConfig) -> None:
        self.limits = limits
        self.prox = prox
        self._ctx: Optional[DrivableSDFContext] = None

        # Active grid built for current project() call (polygon-context mode)
        self._active_grid: Optional[_ActiveGrid] = None
        self._active_key: Optional[Tuple[float, float, float, float, float]] = None
        self._dbg_printed = False

    # -------------------------
    # Context
    # -------------------------
    def set_drivable_sdf_context(self, ctx: Optional[Any]) -> None:
        """Attach an optional drivable-area SDF context.

        Accepts either:
          1) DrivableSDFContext, used by this module directly;
          2) dict returned by navsim.agents.feaxdrive.utils.drivable_sdf.

        The NAVSIM dict rasterizes rows with y_max at row 0. This projector
        samples grids in y_min->y_max order, so dict SDF grids are flipped once
        during conversion.
        """
        if ctx is None:
            self._ctx = None
            return

        if isinstance(ctx, DrivableSDFContext):
            self._ctx = ctx
            return

        if isinstance(ctx, dict):
            if not bool(ctx.get("valid", True)):
                self._ctx = None
                return
            sdf = ctx.get("sdf", None)
            if isinstance(sdf, torch.Tensor):
                # drivable_sdf.py uses image rows: row=0 -> y_max.
                # _grid_sample_sdf uses row=0 -> y_min. Flip if y_max is present.
                if "y_max" in ctx:
                    sdf = torch.flip(sdf, dims=[-2])
            res = float(ctx.get("resolution", ctx.get("res", 0.25)))
            self._ctx = DrivableSDFContext(
                sdf=sdf if isinstance(sdf, torch.Tensor) else None,
                x_min=float(ctx.get("x_min", 0.0)),
                y_min=float(ctx.get("y_min", 0.0)),
                poly_local=ctx.get("poly_local", None),
                resolution=res,
                margin=float(ctx.get("margin", 0.0)),
            )
            return

        # Unknown context type: fail closed as no-op rather than crashing eval.
        self._ctx = None

    @property
    def has_drivable_context(self) -> bool:
        return self._ctx is not None

    # -------------------------
    # Curvature + a_lat metrics
    # -------------------------
    @staticmethod
    def _binomial_kernel(ksize: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Binomial smoothing kernel (Pascal row) as 1D conv weights."""
        if ksize % 2 == 0 or ksize < 3:
            raise ValueError(f"smooth_ksize must be odd and >=3, got {ksize}")
        # closed-form binomial coefficients: C(n, i), n=ksize-1
        n = ksize - 1
        coeffs = []
        # use float computation but stable for small ksize (<=11 typical)
        from math import comb
        for i in range(ksize):
            coeffs.append(float(comb(n, i)))
        coeff = torch.tensor(coeffs, device=device, dtype=dtype)
        coeff = coeff / coeff.sum().clamp_min(1e-12)
        # shape [1,1,ksize] for conv1d
        return coeff.view(1, 1, -1)

    def _smooth_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Apply lightweight 1D smoothing along time dimension."""
        k = int(self.prox.smooth_ksize)
        # Dyn training runs with bf16-mixed precision. CUDA replicate padding
        # does not support bfloat16 in this PyTorch build, so keep the
        # feasibility projector numerics in fp32. The cast is differentiable
        # and gradients are propagated back to the bf16 prediction tensor.
        if xy.dtype in (torch.float16, torch.bfloat16):
            xy = xy.float()
        kernel = self._binomial_kernel(k, xy.device, xy.dtype)
        pad = k // 2
        # [B,2,T]
        xyt = xy.transpose(1, 2)
        xyt = F.pad(xyt, (pad, pad), mode="replicate")
        xyt = F.conv1d(xyt, kernel.expand(2, 1, k), groups=2)
        return xyt.transpose(1, 2)

    def _curvature_and_alat(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute curvature kappa and lateral acceleration a_lat (scheme B).

        Args:
            xy: [B,T,2]
        Returns:
            kappa: [B,T] (endpoints padded with 0)
            a_lat: [B,T]
        """
        if xy.dtype in (torch.float16, torch.bfloat16):
            xy = xy.float()
        B, T, _ = xy.shape
        dt = float(self.limits.dt)
        eps = float(self.prox.eps)

        xy_s = self._smooth_xy(xy)
        # arc length increments
        ds = torch.linalg.norm(xy_s[:, 1:] - xy_s[:, :-1], dim=-1).clamp_min(eps)  # [B,T-1]
        s = torch.cat([torch.zeros((B, 1), device=xy.device, dtype=xy.dtype), torch.cumsum(ds, dim=1)], dim=1)  # [B,T]

        # variable-step central differences for p'(s)
        # denom for i=1..T-2: s[i+1]-s[i-1]
        denom1 = (s[:, 2:] - s[:, :-2]).clamp_min(eps)  # [B,T-2]
        p_prime_mid = (xy_s[:, 2:] - xy_s[:, :-2]) / denom1.unsqueeze(-1)  # [B,T-2,2]

        # variable-step second derivative
        ds_f = (s[:, 2:] - s[:, 1:-1]).clamp_min(eps)   # [B,T-2]
        ds_b = (s[:, 1:-1] - s[:, :-2]).clamp_min(eps)  # [B,T-2]
        term_f = (xy_s[:, 2:] - xy_s[:, 1:-1]) / ds_f.unsqueeze(-1)
        term_b = (xy_s[:, 1:-1] - xy_s[:, :-2]) / ds_b.unsqueeze(-1)
        p_dbl_mid = 2.0 * (term_f - term_b) / denom1.unsqueeze(-1)  # [B,T-2,2]

        # curvature at i=1..T-2
        x1 = p_prime_mid[..., 0]
        y1 = p_prime_mid[..., 1]
        x2 = p_dbl_mid[..., 0]
        y2 = p_dbl_mid[..., 1]
        num = x1 * y2 - y1 * x2
        denom = (x1 * x1 + y1 * y1).clamp_min(eps) ** 1.5
        k_mid = num / denom  # [B,T-2]

        # speed (geometric) and lateral acceleration aligned to mid points
        v_seg = ds / dt  # [B,T-1]
        # align to i=1..T-2 by averaging adjacent segments
        if T >= 3:
            v_mid = 0.5 * (v_seg[:, :-1] + v_seg[:, 1:])  # [B,T-2]
        else:
            v_mid = torch.zeros((B, 0), device=xy.device, dtype=xy.dtype)
        a_lat_mid = (v_mid ** 2) * k_mid

        # pad endpoints
        kappa = torch.zeros((B, T), device=xy.device, dtype=xy.dtype)
        a_lat = torch.zeros((B, T), device=xy.device, dtype=xy.dtype)
        if T >= 3:
            kappa[:, 1:-1] = k_mid
            a_lat[:, 1:-1] = a_lat_mid
        return kappa, a_lat

    def _speed_for_kappa(self, xy: torch.Tensor) -> torch.Tensor:
        """Estimate speed v(t) aligned with curvature time indices.

        We reuse the same arc-length increments as in scheme B:
          v_seg(t) = ||p_{t+1}-p_t|| / dt
        and align to curvature indices (t=1..T-2) by averaging adjacent segments.

        Args:
            xy: [B,T,2]
        Returns:
            v: [B,T] with endpoints padded with 0.
        """
        if xy.dtype in (torch.float16, torch.bfloat16):
            xy = xy.float()
        B, T, _ = xy.shape
        dt = float(self.limits.dt)
        eps = float(self.prox.eps)
        if T < 3:
            return torch.zeros((B, T), device=xy.device, dtype=xy.dtype)

        xy_s = self._smooth_xy(xy)
        ds = torch.linalg.norm(xy_s[:, 1:] - xy_s[:, :-1], dim=-1).clamp_min(eps)  # [B,T-1]
        v_seg = ds / dt  # [B,T-1]
        v_mid = 0.5 * (v_seg[:, :-1] + v_seg[:, 1:])  # [B,T-2]
        v = torch.zeros((B, T), device=xy.device, dtype=xy.dtype)
        v[:, 1:-1] = v_mid
        return v

    def _kappa_max_adapt(self, xy: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute speed-adaptive curvature limit kappa_max_adapt(t).

        If enabled, we use:
            kappa_max_adapt(t) = min(kappa_geo_max, a_lat_max / (v(t)^2 + eps))

        Returns:
            kappa_max_adapt: [B,T] or None if not enabled.
        """
        if xy.dtype in (torch.float16, torch.bfloat16):
            xy = xy.float()
        if not bool(getattr(self.limits, "use_kappa_adapt", False)):
            return None

        # Need at least one of the two bounds.
        k_geo = self.limits.kappa_geo_max
        amax = self.limits.a_lat_max
        if k_geo is None and amax is None:
            return None

        eps = float(self.prox.eps)
        v = self._speed_for_kappa(xy)  # [B,T]
        if amax is None:
            k_dyn = None
        else:
            k_dyn = float(amax) / (v * v + eps)
        if k_geo is None:
            return k_dyn
        k_geo_t = torch.full_like(v, float(k_geo))
        if k_dyn is None:
            return k_geo_t
        return torch.minimum(k_geo_t, k_dyn)

    # -------------------------
    # Penalty + proximal mapping
    # -------------------------
    def violation(self, traj: torch.Tensor, reduction: str = "mean") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Soft constraint penalty + stats for curvature / lateral acceleration."""
        if traj.ndim != 3 or traj.shape[-1] < 2:
            raise ValueError(f"Expected traj [B,T,D>=2], got {traj.shape}")
        work_traj = traj
        if work_traj.dtype in (torch.float16, torch.bfloat16):
            work_traj = work_traj.float()
        xy = work_traj[..., :2]
        kappa, a_lat = self._curvature_and_alat(xy)
        kappa_max_adapt = self._kappa_max_adapt(xy)  # [B,T] or None

        def _relu_sq_abs_excess(val: torch.Tensor, max_abs: Optional[float] = None, max_abs_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Hinge-squared excess over an absolute bound.

            Supports either a scalar bound (max_abs) or a per-time-step tensor bound
            (max_abs_tensor, broadcastable to val).
            """
            if max_abs_tensor is None and max_abs is None:
                return torch.zeros_like(val)
            bound = max_abs_tensor if max_abs_tensor is not None else float(max_abs)  # type: ignore[arg-type]
            excess = torch.relu(val.abs() - bound)
            return excess * excess

        # For curvature, prefer the speed-adaptive bound if enabled.
        if kappa_max_adapt is not None:
            k_pen = _relu_sq_abs_excess(kappa, max_abs_tensor=kappa_max_adapt)
        else:
            k_pen = _relu_sq_abs_excess(kappa, max_abs=self.limits.kappa_max)
        a_pen = _relu_sq_abs_excess(a_lat, self.limits.a_lat_max)

        # per-sample mean
        per = k_pen.mean(dim=1) + a_pen.mean(dim=1)
        if reduction == "mean":
            penalty = per.mean()
        elif reduction == "none":
            penalty = per
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        def _rate(val: torch.Tensor, max_abs: Optional[float] = None, max_abs_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
            if max_abs_tensor is None and max_abs is None:
                return torch.zeros((), device=traj.device, dtype=traj.dtype)
            bound = max_abs_tensor if max_abs_tensor is not None else float(max_abs)  # type: ignore[arg-type]
            return (val.abs() > bound).float().mean()

        stats: Dict[str, torch.Tensor] = {
            "kappa_abs_mean": kappa.abs().mean(),
            "a_lat_abs_mean": a_lat.abs().mean(),
            "kappa_violation_rate": _rate(kappa, self.limits.kappa_max),
            "kappa_adapt_violation_rate": _rate(kappa, max_abs_tensor=kappa_max_adapt),
            "a_lat_violation_rate": _rate(a_lat, self.limits.a_lat_max),
        }
        return penalty, stats


    # -------------------------
    # Footprint utilities
    # -------------------------
    @staticmethod
    def _heading_from_xy(xy: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute heading from xy by finite differences (differentiable)."""
        d = xy[:, 1:] - xy[:, :-1]  # [B,T-1,2]
        hd = torch.atan2(d[..., 1], d[..., 0])
        if h0 is None:
            h0 = hd[:, :1]
        else:
            h0 = h0[:, :1]
        return torch.cat([h0, hd], dim=1)

    @staticmethod
    def _footprint_points(xy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """Return footprint sample points.

        Returns:
            pts: [B,T,N,2]
        """
        mode = os.getenv('DRIVABLE_PROJ_FOOTPRINT_MODE', 'corners').lower()
        lf = float(os.getenv('DRIVABLE_PROJ_LF_M', '3.5'))
        lr = float(os.getenv('DRIVABLE_PROJ_LR_M', '1.0'))
        hw = float(os.getenv('DRIVABLE_PROJ_HALF_WIDTH_M', '1.0'))

        # Always include center for stability
        offsets = [(0.0, 0.0)]

        if mode == 'center':
            pass
        elif mode == 'lr':
            offsets += [(0.0, hw), (0.0, -hw)]
        else:  # corners (default)
            offsets += [
                (lf, hw), (lf, -hw),
                (-lr, hw), (-lr, -hw),
                (lf, 0.0), (-lr, 0.0),
                (0.0, hw), (0.0, -hw),
            ]

        # [N,2]
        off = torch.tensor(offsets, device=xy.device, dtype=xy.dtype)  # (dx,dy) in body frame
        dx = off[:, 0].view(1, 1, -1)
        dy = off[:, 1].view(1, 1, -1)

        c = torch.cos(heading).unsqueeze(-1)
        s = torch.sin(heading).unsqueeze(-1)

        x_off = dx * c - dy * s
        y_off = dx * s + dy * c

        x = xy[..., 0].unsqueeze(-1) + x_off
        y = xy[..., 1].unsqueeze(-1) + y_off
        pts = torch.stack([x, y], dim=-1)  # [B,T,N,2]
        return pts

    @staticmethod
    def _softmin(x: torch.Tensor, tau: float) -> torch.Tensor:
        """Soft-min along last dim."""
        if tau <= 0:
            return x.min(dim=-1).values
        return -tau * torch.logsumexp(-x / tau, dim=-1)

    # -------------------------
    # SDF sampling/build
    # -------------------------
    def _grid_sample_sdf(self, sdf: torch.Tensor, x_min: float, y_min: float, res: float, xy: torch.Tensor) -> torch.Tensor:
        """Sample a fixed SDF grid at xy.

        Args:
            sdf: [1,1,H,W] or [B,1,H,W]
            xy: [B,T,2]
        Returns:
            [B,T]
        """
        if sdf.device != xy.device:
            sdf = sdf.to(device=xy.device)
        if sdf.dtype != xy.dtype:
            sdf = sdf.to(dtype=xy.dtype)

        B, T, _ = xy.shape
        if sdf.shape[0] == 1:
            sdf_b = sdf.expand(B, -1, -1, -1)
        else:
            sdf_b = sdf

        _, _, H, W = sdf_b.shape
        x = xy[..., 0]
        y = xy[..., 1]

        x_norm = (x - x_min) / (res * (W - 1) + 1e-12) * 2.0 - 1.0
        y_norm = (y - y_min) / (res * (H - 1) + 1e-12) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # [B,T,1,2]

        val = F.grid_sample(
            sdf_b,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # [B,1,T,1]
        return val[:, 0, :, 0]

    def _fraction_out_of_grid(self, xy: torch.Tensor) -> float:
        """Fraction of points outside the current active grid."""
        if self._active_grid is None:
            return 0.0
        res = float(self._active_grid.resolution)
        x_min = float(self._active_grid.x_min)
        y_min = float(self._active_grid.y_min)
        H = int(self._active_grid.H)
        W = int(self._active_grid.W)
        x = xy[..., 0]
        y = xy[..., 1]
        x_norm = (x - x_min) / (res * (W - 1) + 1e-12) * 2.0 - 1.0
        y_norm = (y - y_min) / (res * (H - 1) + 1e-12) * 2.0 - 1.0
        out = (x_norm.abs() > 1.0) | (y_norm.abs() > 1.0)
        return float(out.float().mean().item())

    def _build_dynamic_grid_from_polygon(self, poly_local: Any, xy_for_window: torch.Tensor, resolution: float) -> Optional[_ActiveGrid]:
        """Build an SDF grid for a window around the provided xy points."""
        try:
            import numpy as np
            from scipy.ndimage import distance_transform_edt
            from shapely.geometry import box

            # optional shrink
            shrink_m = float(os.getenv('DRIVABLE_PROJ_SHRINK_M', '0.0'))
            if shrink_m > 0:
                poly = poly_local.buffer(-shrink_m)
                if poly.is_empty:
                    poly = poly_local
            else:
                poly = poly_local

            # window bounds
            buf = float(os.getenv('DRIVABLE_PROJ_WINDOW_BUFFER_M', '10.0'))
            xmin = float(xy_for_window[..., 0].min().item()) - buf
            xmax = float(xy_for_window[..., 0].max().item()) + buf
            ymin = float(xy_for_window[..., 1].min().item()) - buf
            ymax = float(xy_for_window[..., 1].max().item()) + buf

            # Ensure sane size
            if xmax <= xmin + 1e-3 or ymax <= ymin + 1e-3:
                return None

            # Clip polygon to window for speed
            win = box(xmin, ymin, xmax, ymax)
            poly = poly.intersection(win)
            if poly.is_empty:
                return None

            W = int(math.floor((xmax - xmin) / resolution)) + 1
            H = int(math.floor((ymax - ymin) / resolution)) + 1
            W = max(W, 8)
            H = max(H, 8)

            xs = xmin + resolution * np.arange(W, dtype=np.float32)
            ys = ymin + resolution * np.arange(H, dtype=np.float32)
            X, Y = np.meshgrid(xs, ys)  # [H,W]

            # Rasterize mask: inside polygon
            mask = None
            try:
                from shapely import vectorized as shp_vect
                mask = shp_vect.contains(poly, X, Y)
            except Exception:
                # Fallback: coarse loop (may be slow). Try to reduce work by using prepared geometry.
                from shapely.prepared import prep
                p = prep(poly)
                mask = np.zeros((H, W), dtype=bool)
                # simple loop over rows (still heavy for huge grids)
                for i in range(H):
                    yi = float(ys[i])
                    # vectorized over x using contains on Points is not available; do small loop
                    for j in range(W):
                        mask[i, j] = p.contains(box(X[i, j], yi, X[i, j], yi))  # degenerate box ~ point

            mask = mask.astype(bool)

            dist_in = distance_transform_edt(mask)
            dist_out = distance_transform_edt(~mask)
            sdf_np = (dist_in - dist_out).astype(np.float32) * float(resolution)

            sdf_t = torch.from_numpy(sdf_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            return _ActiveGrid(sdf=sdf_t, x_min=xmin, y_min=ymin, resolution=float(resolution), H=H, W=W)
        except Exception:
            return None

    def _prepare_active_grid(self, traj_xy: torch.Tensor, heading: torch.Tensor) -> bool:
        """Prepare an active SDF grid for this project() call.

        If fixed-grid context exists, uses it.
        If polygon context exists, builds a dynamic grid around current trajectory.
        """
        self._active_grid = None
        self._active_key = None

        if self._ctx is None:
            return False

        # Fixed-grid takes precedence
        if self._ctx.sdf is not None:
            sdf = self._ctx.sdf
            # ensure shape [1,1,H,W] for sampling helper (we expand B later)
            if sdf.dim() == 4:
                sdf0 = sdf
            else:
                return False
            self._active_grid = _ActiveGrid(
                sdf=sdf0,
                x_min=float(self._ctx.x_min),
                y_min=float(self._ctx.y_min),
                resolution=float(self._ctx.resolution),
                H=int(sdf0.shape[-2]),
                W=int(sdf0.shape[-1]),
            )
            return True

        # Polygon-only mode: build dynamic window
        if self._ctx.poly_local is not None:
            # Use footprint points for window, not only center
            pts = self._footprint_points(traj_xy, heading)  # [B,T,N,2]
            xy_win = pts.reshape(pts.shape[0], -1, 2)
            grid = self._build_dynamic_grid_from_polygon(self._ctx.poly_local, xy_win, float(self._ctx.resolution))
            if grid is None:
                return False
            self._active_grid = grid
            return True

        return False

    def _sample_sdf(self, xy: torch.Tensor) -> Optional[torch.Tensor]:
        """Sample active SDF at xy. Returns [B,T] or None."""
        if self._active_grid is None:
            return None
        return self._grid_sample_sdf(self._active_grid.sdf, self._active_grid.x_min, self._active_grid.y_min, self._active_grid.resolution, xy)

    def _sdf_min_for_traj(self, xy: torch.Tensor, heading: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute footprint-aware minimum SDF along the trajectory."""
        pts = self._footprint_points(xy, heading)  # [B,T,N,2]
        B, T, N, _ = pts.shape
        pts_flat = pts.reshape(B, T * N, 2)
        sdf_flat = self._sample_sdf(pts_flat)
        if sdf_flat is None:
            return None
        sdf_pts = sdf_flat.reshape(B, T, N)
        tau = float(os.getenv('DRIVABLE_PROJ_SOFTMIN_TAU', '0.2'))
        sdf_min = self._softmin(sdf_pts, tau=tau)
        return sdf_min

    # -------------------------
    # Main projection
    # -------------------------
    def project_drivable(self, traj: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        """Project trajectory onto drivable region (approximately)."""
        if traj.ndim != 3 or traj.shape[-1] < 2:
            raise ValueError(f"Expected traj [B,T,D>=2], got {traj.shape}")

        if self._ctx is None:
            if _dbg() and _is_rank0() and not self._dbg_printed:
                self._dbg_printed = True
                print("[drivable_proj] ctx is None -> projection NO-OP (did you inject metric_cache drivable context?)")
            return traj


        z = traj[..., :2]

        # Heading (if provided); otherwise derive from z
        h0 = traj[..., 2] if traj.shape[-1] >= 3 else None
        heading0 = self._heading_from_xy(z, h0=h0)

        # Prepare SDF grid for this call
        if not self._prepare_active_grid(z, heading0):
            if _dbg() and _is_rank0() and not self._dbg_printed:
                self._dbg_printed = True
                print("[drivable_proj] prepare_active_grid failed -> projection NO-OP (SDF window/coords issue)")
            return traj


        # Out-of-grid fallback (fixed-grid SDF mode): if many footprint points fall outside
        # the precomputed SDF window, border sampling can make projection ineffective.
        # When polygon is available, rebuild a dynamic SDF window around this trajectory.
        if self._ctx.sdf is not None and self._ctx.poly_local is not None:
            with torch.no_grad():
                pts0 = self._footprint_points(z, heading0)  # [B,T,N,2]
                B, T, N, _ = pts0.shape
                pts0_flat = pts0.reshape(B, T * N, 2)
                oog = self._fraction_out_of_grid(pts0_flat)
                thresh = float(os.getenv('DRIVABLE_PROJ_OOG_FALLBACK', '0.05'))
                if oog > thresh:
                    grid = self._build_dynamic_grid_from_polygon(self._ctx.poly_local, pts0_flat, float(self._ctx.resolution))
                    if grid is not None:
                        self._active_grid = grid
                        if _dbg() and _is_rank0():
                            print(f"[drivable_proj] out-of-grid fallback: oog={oog:.3f} > {thresh:.3f}, built dynamic grid")

        # Effective margin for triggering (cap huge margins)
        margin_raw = float(getattr(self._ctx, 'margin', 0.0))
        cap_m = float(os.getenv('DRIVABLE_PROJ_MARGIN_CAP_M', '1.0'))
        margin_eff = margin_raw if cap_m <= 0 else min(margin_raw, cap_m)

        # Early exit: if entire footprint already safe
        with torch.no_grad():
            sdf_min0 = self._sdf_min_for_traj(z, heading0)
            if sdf_min0 is None:
                if _dbg() and _is_rank0() and not self._dbg_printed:
                    self._dbg_printed = True
                    print("[drivable_proj] sdf_min0 is None -> projection NO-OP (SDF sampling failed / out-of-grid / missing polygon)")
                return traj
            # Print trigger stats once
            if _dbg() and _is_rank0() and not self._dbg_printed:
                self._dbg_printed = True
                trig = (sdf_min0 < margin_eff).float().mean().item()
                print(f"[drivable_proj] margin_raw={margin_raw:.3f}  margin_eff={margin_eff:.3f}  trigger_rate={trig:.3f}  sdf_min={float(sdf_min0.min()):.3f}  sdf_mean={float(sdf_min0.mean()):.3f}")

            if torch.all(sdf_min0 >= margin_eff):
                return traj

        steps = max(int(self.prox.steps), 1)
        eta = float(self.prox.step_size)
        lam = float(self.prox.lambda_kappa)  # drivable weight

        # Optional stronger penalty for outside points
        lam_out = float(os.getenv('DRIVABLE_PROJ_LAMBDA_OUT', str(lam)))

        xy = z

        for _ in range(steps):
            xy_in = xy if create_graph else xy.detach()

            with torch.enable_grad():
                xy_var = xy_in.clone().requires_grad_(True)
                # update heading based on current xy (so footprint orientation follows trajectory)
                h_var = self._heading_from_xy(xy_var, h0=h0)

                sdf_min = self._sdf_min_for_traj(xy_var, h_var)
                if sdf_min is None:
                    break

                excess = torch.relu(margin_eff - sdf_min)
                outside = torch.relu(-sdf_min)
                phi = lam * (excess * excess).mean() + lam_out * (outside * outside).mean()

                obj = 0.5 * ((xy_var - z) ** 2).mean() + phi
                (grad,) = torch.autograd.grad(obj, xy_var, create_graph=create_graph, retain_graph=False)

                xy_next = xy_var - eta * grad

            if not create_graph:
                xy_next = xy_next.detach()

            if bool(self.prox.keep_start):
                xy_next = torch.cat([z[:, :1].detach(), xy_next[:, 1:]], dim=1)

            xy = xy_next

            # optional early stop
            with torch.no_grad():
                h_tmp = self._heading_from_xy(xy, h0=h0)
                sdf_tmp = self._sdf_min_for_traj(xy, h_tmp)
                if sdf_tmp is None:
                    break
                if torch.all(sdf_tmp >= margin_eff):
                    break

        out = traj.clone()
        out[..., :2] = xy
        if out.shape[-1] >= 3:
            h = self._heading_from_xy(xy, h0=out[..., 2])
            out[..., 2] = h

        # Clear active grid to avoid accidentally reusing across scenes
        self._active_grid = None
        self._active_key = None
        self._dbg_printed = False

        return out

    def project_dynamics(self, traj: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        """Proximal projection focusing on curvature / lateral acceleration.

        Args:
            traj: [B,T,D] (x,y,heading,...)
            create_graph: whether to build higher-order graph (for training)
        """
        if traj.ndim != 3 or traj.shape[-1] < 2:
            raise ValueError(f"Expected traj [B,T,D>=2], got {traj.shape}")

        # If no curvature-related limits are set, return as-is.
        # Note: when use_kappa_adapt=True, we may rely on kappa_geo_max and/or a_lat_max.
        if (
            self.limits.kappa_max is None
            and self.limits.a_lat_max is None
            and not bool(getattr(self.limits, "use_kappa_adapt", False))
        ):
            return traj

        z = traj[..., :2]
        xy = z

        steps = max(int(self.prox.steps), 1)
        eta = float(self.prox.step_size)
        lam_k = float(self.prox.lambda_kappa)
        lam_a = float(self.prox.lambda_a_lat)

        # unrolled proximal GD
        for _ in range(steps):
            # 关键：val / 推理常在 torch.no_grad()，这里必须显式 enable_grad 才能做近端梯度
            xy_in = xy if create_graph else xy.detach()

            with torch.enable_grad():
                xy_var = xy_in.clone().requires_grad_(True)  # [B,T,2]

                kappa, a_lat = self._curvature_and_alat(xy_var)
                kappa_max_adapt = self._kappa_max_adapt(xy_var)

                # hinge-squared penalties
                phi = torch.zeros((), device=traj.device, dtype=traj.dtype)
                # Curvature penalty: prefer speed-adaptive bound if enabled.
                if kappa_max_adapt is not None:
                    excess_k = torch.relu(kappa.abs() - kappa_max_adapt)
                    phi = phi + lam_k * (excess_k * excess_k).mean()
                elif self.limits.kappa_max is not None:
                    excess_k = torch.relu(kappa.abs() - float(self.limits.kappa_max))
                    phi = phi + lam_k * (excess_k * excess_k).mean()
                if self.limits.a_lat_max is not None:
                    excess_a = torch.relu(a_lat.abs() - float(self.limits.a_lat_max))
                    phi = phi + lam_a * (excess_a * excess_a).mean()

                # proximal objective
                obj = 0.5 * ((xy_var - z) ** 2).mean() + phi

                (grad,) = torch.autograd.grad(
                    obj, xy_var,
                    create_graph=create_graph,
                    retain_graph=False
                )

                xy_next = xy_var - eta * grad

            # inference/val：别把图带出循环，防止显存/图堆积；train(create_graph=True)则保留图
            if not create_graph:
                xy_next = xy_next.detach()


            # keep the start pose fixed (anchor)
            if bool(self.prox.keep_start):
                xy_next = torch.cat([z[:, :1].detach(), xy_next[:, 1:]], dim=1)

            xy = xy_next

        out = traj.clone()
        out[..., :2] = xy
        if out.shape[-1] >= 3:
            d = xy[:, 1:] - xy[:, :-1]                 # [B,T-1,2]
            hd = torch.atan2(d[..., 1], d[..., 0])     # [B,T-1]
            # pad to length T (keep first heading from original, or use first hd)
            h0 = out[:, :1, 2]                         # [B,1]
            h = torch.cat([h0, hd], dim=1)             # [B,T]
            out[..., 2] = h
        return out


    def project(self, traj: torch.Tensor, *, create_graph: bool = False, mode: str = "auto") -> torch.Tensor:
        """Unified projection entry.

        Args:
            traj: [B,T,D>=2] trajectory in metric space.
            create_graph: keep higher-order graph for training-time projection losses.
            mode:
              - "dynamics"/"dyn"/"curvature": v4 curvature + lateral-acceleration projector.
              - "drivable"/"drive"/"guidance": v4drive footprint-level drivable projector.
              - "auto" (default): use drivable projector only when a drivable context has
                been attached; otherwise use dynamics projector.

        This preserves the historical behavior:
          - training-time dyn regularization calls violation() and/or project() without
            a drivable context, so it uses the v4 dynamics projector;
          - guidance/inference code that attaches an SDF context can use the v4drive
            drivable projector through the same class.
        """
        m = str(mode or "auto").lower()
        if m in {"dynamics", "dynamic", "dyn", "curvature", "kinematic", "kinematics"}:
            return self.project_dynamics(traj, create_graph=create_graph)
        if m in {"drivable", "drive", "guidance", "sdf", "map"}:
            return self.project_drivable(traj, create_graph=create_graph)
        if m != "auto":
            raise ValueError(f"Unsupported projection mode: {mode}")
        if self.has_drivable_context:
            return self.project_drivable(traj, create_graph=create_graph)
        return self.project_dynamics(traj, create_graph=create_graph)


# -------------------------
# Optional: build polygon-based context from metric_cache
# (If your pipeline already builds fixed-grid SDF, you can ignore this.)
# -------------------------

def build_drivable_polygon_context_from_metric_cache(metric_cache: Any, *, resolution: float, margin: float) -> Optional[DrivableSDFContext]:
    """Build a polygon-only drivable context in ego-local coordinates.

    This enables dynamic SDF window build in the projector.

    Requirements:
      - metric_cache.drivable_area_map provides drivable polygons
      - metric_cache.ego_state provides rear_axle pose (x,y,yaw)

    Returns:
      DrivableSDFContext with poly_local filled (and sdf=None)
    """
    try:
        from shapely.ops import unary_union, transform
        import numpy as np

        dam = getattr(metric_cache, 'drivable_area_map', None)
        if dam is None:
            return None

        geoms = None
        # Try common attribute names
        for name in ('_geometries', 'geometries', 'polygons', 'geometry'):
            if hasattr(dam, name):
                geoms = getattr(dam, name)
                break
        if geoms is None:
            return None

        # geoms may be a list or dict
        if isinstance(geoms, dict):
            geom_list = list(geoms.values())
        else:
            geom_list = list(geoms)
        if len(geom_list) == 0:
            return None

        poly = unary_union(geom_list)

        ego = getattr(metric_cache, 'ego_state', None)
        if ego is None:
            return None
        rear = getattr(ego, 'rear_axle', None)
        if rear is None:
            return None
        x0 = float(getattr(rear, 'x'))
        y0 = float(getattr(rear, 'y'))
        yaw = float(getattr(rear, 'heading'))

        c = math.cos(-yaw)
        s = math.sin(-yaw)

        def _to_local(x, y, z=None):
            dx = x - x0
            dy = y - y0
            xl = c * dx - s * dy
            yl = s * dx + c * dy
            return (xl, yl)

        poly_local = transform(_to_local, poly)

        return DrivableSDFContext(
            sdf=None,
            x_min=0.0,
            y_min=0.0,
            poly_local=poly_local,
            resolution=float(resolution),
            margin=float(margin),
        )
    except Exception:
        return None


def build_drivable_sdf_context_from_metric_cache(
    metric_cache: Any,
    *,
    resolution: float,
    margin: float,
    x_min: float = -1.57,
    x_max: float = 65.17,
    y_min: float = -19.68,
    y_max: float = 22.32,
) -> Optional[DrivableSDFContext]:
    """Build a fixed-grid SDF drivable context from metric_cache.

    This keeps backward compatibility with earlier integration code that expects
    a precomputed SDF grid.

    Args:
        metric_cache: scenario metric cache (must contain drivable_area_map and ego_state.rear_axle)
        resolution: meters per pixel
        margin: clearance threshold in meters (stored in ctx; projection may cap it for triggering)
        x_min/x_max/y_min/y_max: local ego-frame grid bounds (meters)

    Returns:
        DrivableSDFContext or None on failure.
    """
    try:
        import numpy as np
        from scipy.ndimage import distance_transform_edt
        from shapely.ops import unary_union, transform

        dam = getattr(metric_cache, 'drivable_area_map', None)
        if dam is None:
            return None

        geoms = None
        for name in ('_geometries', 'geometries', 'polygons', 'geometry'):
            if hasattr(dam, name):
                geoms = getattr(dam, name)
                break
        if geoms is None:
            return None

        if isinstance(geoms, dict):
            geom_list = list(geoms.values())
        else:
            geom_list = list(geoms)
        if len(geom_list) == 0:
            return None

        poly_world = unary_union(geom_list)

        ego = getattr(metric_cache, 'ego_state', None)
        if ego is None:
            return None
        rear = getattr(ego, 'rear_axle', None)
        if rear is None:
            return None
        x0 = float(getattr(rear, 'x'))
        y0 = float(getattr(rear, 'y'))
        yaw = float(getattr(rear, 'heading'))

        c = math.cos(-yaw)
        s = math.sin(-yaw)

        def _to_local(x, y, z=None):
            dx = x - x0
            dy = y - y0
            xl = c * dx - s * dy
            yl = s * dx + c * dy
            return (xl, yl)

        poly_local = transform(_to_local, poly_world)

        # Optional shrink to compensate for footprint metrics (configurable)
        shrink_m = float(os.getenv('DRIVABLE_PROJ_SHRINK_M', '0.0'))
        if shrink_m > 0:
            poly_s = poly_local.buffer(-shrink_m)
            if not poly_s.is_empty:
                poly_local = poly_s

        W = int(math.floor((x_max - x_min) / resolution)) + 1
        H = int(math.floor((y_max - y_min) / resolution)) + 1
        W = max(W, 8)
        H = max(H, 8)

        xs = x_min + resolution * np.arange(W, dtype=np.float32)
        ys = y_min + resolution * np.arange(H, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        mask = None
        try:
            from shapely import vectorized as shp_vect
            mask = shp_vect.contains(poly_local, X, Y)
        except Exception:
            from shapely.prepared import prep
            from shapely.geometry import Point
            p = prep(poly_local)
            mask = np.zeros((H, W), dtype=bool)
            # fallback: row-wise loops (slow for large grids)
            for i in range(H):
                yi = float(ys[i])
                for j in range(W):
                    mask[i, j] = p.contains(Point(float(xs[j]), yi))

        mask = mask.astype(bool)
        dist_in = distance_transform_edt(mask)
        dist_out = distance_transform_edt(~mask)
        sdf_np = (dist_in - dist_out).astype(np.float32) * float(resolution)

        sdf_t = torch.from_numpy(sdf_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        if _dbg() and _is_rank0():
            try:
                print('[drivable_proj][SDF BUILD] resolution(m/px)=', float(resolution), ' margin(m)=', float(margin))
                print('[drivable_proj][SDF BUILD] grid x:[', float(x_min), ',', float(x_max), '] y:[', float(y_min), ',', float(y_max), ']')
                print('[drivable_proj][SDF BUILD] dist_in px max=', float(dist_in.max()), ' dist_out px max=', float(dist_out.max()))
                print('[drivable_proj][SDF BUILD] sdf_m max≈', float(sdf_np.max()), 'm')
            except Exception:
                pass

        return DrivableSDFContext(
            sdf=sdf_t,
            x_min=float(x_min),
            y_min=float(y_min),
            poly_local=poly_local,
            resolution=float(resolution),
            margin=float(margin),
        )

    except Exception:
        return None
