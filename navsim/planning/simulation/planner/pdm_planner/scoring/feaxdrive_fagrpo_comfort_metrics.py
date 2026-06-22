"""FA-GRPO comfort / feasibility metrics for FeaXDrive training reward.

This module is intentionally separate from ``pdm_comfort_metrics.py`` used
by final standard PDM evaluation. It adds curvature feasibility checks used
only by FA-GRPO reward scoring.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

# TODO: Refactor & add to config

# (1) ego_jerk_metric,
max_abs_mag_jerk = 8.37  # [m/s^3]

# (2) ego_lat_acceleration_metric
max_abs_lat_accel = 4.89  # [m/s^2]

# (3) ego_lon_acceleration_metric
max_lon_accel = 2.40  # [m/s^2]
min_lon_accel = -4.05

# (4) ego_yaw_acceleration_metric
max_abs_yaw_accel = 1.93  # [rad/s^2]

# (5) ego_lon_jerk_metric
max_abs_lon_jerk = 4.13  # [m/s^3]

# (6) ego_yaw_rate_metric
max_abs_yaw_rate = 0.95  # [rad/s]

# (7) curvature compliance metric (binary, same aggregation form as comfort)
# Use the same feasibility thresholds as the prior dyn evaluation setup.
max_abs_curvature_geo = 0.166  # [1/m]
max_abs_lat_accel_for_curvature = 6.0  # [m/s^2]
curvature_smooth_ksize = 5
curvature_eps = 1e-6


def _extract_ego_acceleration(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float32]:
    """
    Extract acceleration of ego pose in simulation history over batch-dim
    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param decimals: decimal precision, defaults to 8
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 8
    :raises ValueError: when coordinate not available
    :return: array containing acceleration values
    """

    n_batch, n_time, n_states = states.shape
    if acceleration_coordinate == "x":
        acceleration: npt.NDArray[np.float64] = states[..., StateIndex.ACCELERATION_X]

    elif acceleration_coordinate == "y":
        acceleration: npt.NDArray[np.float64] = states[..., StateIndex.ACCELERATION_Y]

    elif acceleration_coordinate == "magnitude":
        acceleration: npt.NDArray[np.float64] = np.hypot(
            states[..., StateIndex.ACCELERATION_X],
            states[..., StateIndex.ACCELERATION_Y],
        )
    else:
        raise ValueError(
            f"acceleration_coordinate option: {acceleration_coordinate} not available. "
            f"Available options are: x, y or magnitude"
        )

    acceleration = savgol_filter(
        acceleration,
        polyorder=poly_order,
        window_length=min(window_length, n_time),
        axis=-1,
    )
    acceleration = np.round(acceleration, decimals=decimals)
    return acceleration


def _extract_ego_jerk(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    time_steps_s: npt.NDArray[np.float64],
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract jerk of ego pose in simulation history over batch-dim
    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param time_steps_s: time steps [s] of time dim
    :param decimals: decimal precision, defaults to 8
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 15
    :return: array containing jerk values
    """
    n_batch, n_time, n_states = states.shape
    ego_acceleration = _extract_ego_acceleration(states, acceleration_coordinate=acceleration_coordinate)
    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    jerk = np.round(jerk, decimals=decimals)
    return jerk


def _extract_ego_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract yaw-rate of simulation history over batch-dim
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param decimals:  decimal precision, defaults to 8
    :param window_length: window size for extraction, defaults to 15
    :return: array containing ego's yaw rate
    """
    ego_headings = states[..., StateIndex.HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
    )  # convert to seconds
    ego_yaw_rate = np.round(ego_yaw_rate, decimals=decimals)
    return ego_yaw_rate


def _phase_unwrap(headings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Returns an array of heading angles equal mod 2 pi to the input heading angles,
    and such that the difference between successive output angles is less than or
    equal to pi radians in absolute value
    :param headings: An array of headings (radians)
    :return The phase-unwrapped equivalent headings.
    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation of yaw to be very large.
    # We want unwrapped[j] = headings[j] - 2*pi*adjustments[j] for some integer-valued adjustments making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= headings[j+1] - headings[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (headings[j+1] - headings[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((headings[j+1] - headings[j]) / (2*pi)).
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    unwrapped = headings - two_pi * adjustments
    return unwrapped


def _approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx,
        axis=axis,
    )
    return derivative


def _within_bound(
    metric: npt.NDArray[np.float64],
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> npt.NDArray[np.bool_]:
    """
    Determines wether values in batch-dim are within bounds.
    :param metric: metric values
    :param min_bound: minimum bound, defaults to None
    :param max_bound: maximum bound, defaults to None
    :return: array of booleans wether metric values are within bounds
    """
    min_bound = min_bound if min_bound else float(-np.inf)
    max_bound = max_bound if max_bound else float(np.inf)
    metric_values = np.array(metric)
    metric_within_bound = (metric_values > min_bound) & (metric_values < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _binomial_kernel(ksize: int) -> npt.NDArray[np.float64]:
    """Binomial (Pascal) smoothing kernel for curvature estimation."""
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError(f"curvature_smooth_ksize must be odd and >=3, got {ksize}")
    from math import comb

    n = ksize - 1
    coeff = np.array([comb(n, i) for i in range(ksize)], dtype=np.float64)
    coeff = coeff / np.sum(coeff)
    return coeff


def _smooth_xy_batch(xy: npt.NDArray[np.float64], kernel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Replicate-padding smoothing along the time axis for batched x/y sequences."""
    pad = len(kernel) // 2
    if pad <= 0:
        return xy
    left = np.repeat(xy[:, :1, :], pad, axis=1)
    right = np.repeat(xy[:, -1:, :], pad, axis=1)
    padded = np.concatenate([left, xy, right], axis=1)
    out = np.zeros_like(xy, dtype=np.float64)
    for i in range(xy.shape[1]):
        window = padded[:, i : i + len(kernel), :]
        out[:, i, :] = np.sum(window * kernel[None, :, None], axis=1)
    return out


def _estimate_curvature_and_speed(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    eps: float = curvature_eps,
    smooth_ksize: int = curvature_smooth_ksize,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Estimate arc-length curvature and aligned speed for simulated states.

    Returns:
        kappa: [B, T] curvature with zero-padded endpoints
        v: [B, T] speed aligned to curvature indices with zero-padded endpoints
    """
    xy = states[..., StateIndex.POINT].astype(np.float64)
    n_batch, n_time, _ = xy.shape
    if n_time < 3:
        return np.zeros((n_batch, n_time), dtype=np.float64), np.zeros((n_batch, n_time), dtype=np.float64)

    kernel = _binomial_kernel(smooth_ksize)
    xy_s = _smooth_xy_batch(xy, kernel)

    ds = np.linalg.norm(xy_s[:, 1:] - xy_s[:, :-1], axis=-1)
    ds = np.maximum(ds, eps)
    s = np.concatenate([np.zeros((n_batch, 1), dtype=np.float64), np.cumsum(ds, axis=1)], axis=1)

    denom1 = np.maximum(s[:, 2:] - s[:, :-2], eps)
    p_prime = (xy_s[:, 2:] - xy_s[:, :-2]) / denom1[..., None]

    ds_f = np.maximum(s[:, 2:] - s[:, 1:-1], eps)
    ds_b = np.maximum(s[:, 1:-1] - s[:, :-2], eps)
    term_f = (xy_s[:, 2:] - xy_s[:, 1:-1]) / ds_f[..., None]
    term_b = (xy_s[:, 1:-1] - xy_s[:, :-2]) / ds_b[..., None]
    p_dbl = 2.0 * (term_f - term_b) / denom1[..., None]

    x1, y1 = p_prime[..., 0], p_prime[..., 1]
    x2, y2 = p_dbl[..., 0], p_dbl[..., 1]
    num = x1 * y2 - y1 * x2
    denom = np.power(np.maximum(x1 * x1 + y1 * y1, eps), 1.5)
    k_mid = num / denom

    # speed aligned to curvature midpoints, consistent with the prior dyn metric.
    dt = float(np.mean(np.diff(time_steps_s))) if len(time_steps_s) > 1 else 0.1
    dt = max(dt, eps)
    v_seg = ds / dt
    v_mid = 0.5 * (v_seg[:, :-1] + v_seg[:, 1:])

    kappa = np.zeros((n_batch, n_time), dtype=np.float64)
    v = np.zeros((n_batch, n_time), dtype=np.float64)
    kappa[:, 1:-1] = k_mid
    v[:, 1:-1] = v_mid
    return kappa, v


def _compute_curvature_metric(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Binary speed-adaptive curvature compliance, aggregated like comfort.

    A proposal is marked comfortable w.r.t. curvature only if every time step satisfies:
        |kappa(t)| < min(kappa_geo_max, a_lat_max / (v(t)^2 + eps)).
    """
    n_batch, n_time, n_states = states.shape
    kappa, v = _estimate_curvature_and_speed(states, time_steps_s=time_steps_s)
    kappa_bound = np.minimum(
        np.full_like(kappa, max_abs_curvature_geo, dtype=np.float64),
        max_abs_lat_accel_for_curvature / (v * v + curvature_eps),
    )
    return np.all(np.abs(kappa) < kappa_bound, axis=-1)


def _compute_lon_acceleration(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute longitudinal acceleration over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: longitudinal acceleration within bound
    """
    n_batch, n_time, n_states = states.shape
    lon_acceleration = _extract_ego_acceleration(states, acceleration_coordinate="x", window_length=n_time)
    return _within_bound(lon_acceleration, min_bound=min_lon_accel, max_bound=max_lon_accel)


def _compute_lat_acceleration(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute lateral acceleration over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: lateral acceleration within bound
    """
    n_batch, n_time, n_states = states.shape
    lat_acceleration = _extract_ego_acceleration(states, acceleration_coordinate="y", window_length=n_time)
    return _within_bound(lat_acceleration, min_bound=-max_abs_lat_accel, max_bound=max_abs_lat_accel)


def _compute_jerk_metric(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute absolute jerk over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: absolute jerk within bound
    """
    n_batch, n_time, n_states = states.shape
    jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="magnitude",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )
    return _within_bound(jerk_metric, min_bound=-max_abs_mag_jerk, max_bound=max_abs_mag_jerk)


def _compute_lon_jerk_metric(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute longitudinal jerk over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: longitudinal jerk within bound
    """
    n_batch, n_time, n_states = states.shape
    lon_jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="x",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )
    return _within_bound(lon_jerk_metric, min_bound=-max_abs_lon_jerk, max_bound=max_abs_lon_jerk)


def _compute_yaw_accel(states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Compute acceleration of yaw-angle over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: acceleration of yaw-angle within bound
    """
    n_batch, n_time, n_states = states.shape
    yaw_accel_metric = _extract_ego_yaw_rate(states, time_steps_s, deriv_order=2, poly_order=3, window_length=n_time)
    return _within_bound(yaw_accel_metric, min_bound=-max_abs_yaw_accel, max_bound=max_abs_yaw_accel)


def _compute_yaw_rate(states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Compute velocity of yaw-angle over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: velocity of yaw-angle within bound
    """
    n_batch, n_time, n_states = states.shape
    yaw_rate_metric = _extract_ego_yaw_rate(states, time_steps_s, window_length=n_time)
    return _within_bound(yaw_rate_metric, min_bound=-max_abs_yaw_rate, max_bound=max_abs_yaw_rate)


def ego_is_comfortable(states: npt.NDArray[np.float64], time_point_s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Accumulates all within-bound comfortability metrics
    :param states: array representation of ego state values
    :param time_point_s: time steps [s] of time dim
    :return: _description_
    """
    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_point_s)
    assert n_states == StateIndex.size()

    comfort_metric_functions = [
        _compute_lon_acceleration,
        _compute_lat_acceleration,
        _compute_jerk_metric,
        _compute_lon_jerk_metric,
        _compute_yaw_accel,
        _compute_yaw_rate,
        _compute_curvature_metric,
    ]
    results: npt.NDArray[np.bool_] = np.zeros((n_batch, len(comfort_metric_functions)), dtype=np.bool_)
    for idx, metric_function in enumerate(comfort_metric_functions):
        results[:, idx] = metric_function(states, time_point_s)

    return results
