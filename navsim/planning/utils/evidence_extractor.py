from __future__ import annotations

"""Unified EvidenceExtractor for CFQA generation and inference.

Design goals:
- Same evidence schema for offline dataset generation and online inference.
- Lightweight: no heavy models required.
- Supports corridor-based object relevance with a provided polyline.

Evidence groups (lightweight):
1) Object relevance: cone/ped/bicycle relative to a corridor polyline.
2) Visibility score: contrast+sharpness+glare heuristics from the image.
3) Lane confidence score: edge/line heuristics from the image (lower half).

Notes:
- Corridor polyline should be in ego frame coordinates (x forward, y left), as a list of (x,y).
- For offline generation, you may pass GT future trajectory polyline as corridor.
- For online inference, pass route/map centerline corridor.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _polyline_min_dist(point_xy: Tuple[float, float], polyline_xy: Sequence[Tuple[float, float]]) -> float:
    """Minimum Euclidean distance from point to polyline."""
    px, py = point_xy
    best = 1e9
    if polyline_xy is None or len(polyline_xy) < 2:
        return best
    for i in range(len(polyline_xy) - 1):
        x1, y1 = polyline_xy[i]
        x2, y2 = polyline_xy[i + 1]
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        seg_len2 = vx * vx + vy * vy + 1e-9
        t = (wx * vx + wy * vy) / seg_len2
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        cx, cy = x1 + t * vx, y1 + t * vy
        d = math.hypot(px - cx, py - cy)
        if d < best:
            best = d
    return best


def _speed_bin(speed_mps: float) -> str:
    if speed_mps < 2.0:
        return "low"
    if speed_mps < 7.0:
        return "mid"
    return "high"


@dataclass
class ObjRel:
    has_obj: bool
    rel: str  # none/off_path/near_path/on_path
    min_obj_dist_m: Optional[float]
    min_path_dist_m: Optional[float]


class EvidenceExtractor:
    def __init__(
        self,
        corridor_half_width_m: float = 1.8,
        near_margin_m: float = 1.2,
        lookahead_m: float = 30.0,
    ) -> None:
        self.corridor_half_width_m = float(corridor_half_width_m)
        self.near_margin_m = float(near_margin_m)
        self.lookahead_m = float(lookahead_m)

    # ---------------------
    # Object relevance
    # ---------------------
    def class_relevance(
        self,
        obj_names: Sequence[str],
        obj_xy: np.ndarray,
        class_name: str,
        corridor_polyline_xy: Sequence[Tuple[float, float]],
    ) -> ObjRel:
        if obj_names is None or len(obj_names) == 0 or obj_xy is None or len(obj_xy) == 0:
            return ObjRel(False, "none", None, None)

        ego_dists: List[float] = []
        path_dists: List[float] = []

        for name, box in zip(obj_names, obj_xy):
            if str(name) != class_name:
                continue
            x = float(box[0]); y = float(box[1])
            # ignore behind
            if x < -2.0:
                continue
            ego_dist = float(math.hypot(x, y))
            if ego_dist > self.lookahead_m:
                continue
            path_dist = float(_polyline_min_dist((x, y), corridor_polyline_xy))
            ego_dists.append(ego_dist)
            path_dists.append(path_dist)

        if not ego_dists:
            return ObjRel(False, "none", None, None)

        min_ego = float(min(ego_dists))
        min_path = float(min(path_dists))

        if min_path <= self.corridor_half_width_m:
            rel = "on_path"
        elif min_path <= self.corridor_half_width_m + self.near_margin_m:
            rel = "near_path"
        else:
            rel = "off_path"

        return ObjRel(True, rel, min_ego, min_path)

    def nearest_object_dist(self, obj_xy: np.ndarray) -> Optional[float]:
        if obj_xy is None or len(obj_xy) == 0:
            return None
        d = float(np.min(np.hypot(obj_xy[:, 0], obj_xy[:, 1])))
        return d if math.isfinite(d) else None

    # ---------------------
    # Image heuristics
    # ---------------------
    def visibility_score(self, image_path: str) -> float:
        """Return visibility score in [0,1], higher is better."""
        if cv2 is None:
            return 0.5
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return 0.5
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Contrast (std)
        contrast = float(np.std(gray) / 64.0)  # ~0..2
        contrast = max(0.0, min(1.0, contrast))

        # Sharpness (Laplacian variance)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharp = float(lap.var() / 300.0)  # heuristic
        sharp = max(0.0, min(1.0, sharp))

        # Glare/overexposure: fraction of very bright pixels
        bright_frac = float(np.mean(gray > 245))
        glare = max(0.0, min(1.0, bright_frac * 5.0))

        # Fog-like low contrast + low sharpness indicator
        fog_penalty = (1.0 - contrast) * (1.0 - sharp)

        score = 0.55 * contrast + 0.45 * sharp
        score = score * (1.0 - 0.6 * glare) * (1.0 - 0.4 * fog_penalty)
        return float(max(0.0, min(1.0, score)))

    def lane_conf_score(self, image_path: str) -> float:
        """Return lane confidence score in [0,1], higher means clearer lane/boundary cues.

        This is a lightweight heuristic using edges/lines in the lower half.
        """
        if cv2 is None:
            return 0.5
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return 0.5
        h, w = img.shape[:2]
        roi = img[int(h * 0.5):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Edge density
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.mean(edges > 0))  # 0..1

        # Line evidence
        line_score = 0.0
        if w > 0:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=int(w * 0.08), maxLineGap=20)
            if lines is not None:
                angles = []
                lengths = []
                for x1, y1, x2, y2 in lines[:, 0, :]:
                    dx = x2 - x1
                    dy = y2 - y1
                    length = math.hypot(dx, dy)
                    if length < 20:
                        continue
                    angle = abs(math.degrees(math.atan2(dy, dx)))
                    angles.append(angle)
                    lengths.append(length)
                if lengths:
                    # lanes tend to have near-vertical-ish lines in image ROI (depends on camera)
                    # Accept both 20-80 degrees as weak evidence.
                    good = [l for a, l in zip(angles, lengths) if 20 <= a <= 80]
                    line_score = float(min(1.0, (sum(good) / (sum(lengths) + 1e-6))))

        # Combine
        # edge_density too high may indicate texture/noise, so squash.
        ed = max(0.0, min(1.0, edge_density * 4.0))
        score = 0.6 * ed + 0.4 * line_score
        return float(max(0.0, min(1.0, score)))

    # ---------------------
    # Full extraction
    # ---------------------
    def extract(
        self,
        image_path: str,
        ego_velocity_xy: Sequence[float],
        nav_cmd_onehot: Optional[Sequence[int]],
        obj_names: Sequence[str],
        obj_boxes_xyzlwhh: np.ndarray,
        corridor_polyline_xy: Sequence[Tuple[float, float]],
    ) -> Dict[str, Any]:
        speed = float(math.hypot(float(ego_velocity_xy[0]), float(ego_velocity_xy[1])))
        e: Dict[str, Any] = {
            "ego_speed_mps": speed,
            "ego_speed_bin": _speed_bin(speed),
            "nav_cmd": self._nav_cmd_from_onehot(nav_cmd_onehot),
        }

        obj_xy = obj_boxes_xyzlwhh[:, :2] if obj_boxes_xyzlwhh is not None and len(obj_boxes_xyzlwhh) > 0 else np.zeros((0, 2), dtype=np.float32)
        e["dense_objects"] = bool(len(obj_xy) >= 12)
        e["min_obj_dist"] = self.nearest_object_dist(obj_xy)

        ped = self.class_relevance(obj_names, obj_xy, "pedestrian", corridor_polyline_xy)
        bic = self.class_relevance(obj_names, obj_xy, "bicycle", corridor_polyline_xy)
        cone = self.class_relevance(obj_names, obj_xy, "traffic_cone", corridor_polyline_xy)

        e.update({
            "ped_rel": ped.rel,
            "bicycle_rel": bic.rel,
            "cone_rel": cone.rel,
            "min_ped_obj_dist": ped.min_obj_dist_m,
            "min_ped_path_dist": ped.min_path_dist_m,
            "min_bicycle_obj_dist": bic.min_obj_dist_m,
            "min_bicycle_path_dist": bic.min_path_dist_m,
            "min_cone_obj_dist": cone.min_obj_dist_m,
            "min_cone_path_dist": cone.min_path_dist_m,
        })

        # Lightweight image evidence
        e["visibility_score"] = self.visibility_score(image_path)
        e["lane_conf"] = self.lane_conf_score(image_path)

        # Proxies (optional, can be refined later)
        # Construction score uses cone relevance (on/near) as proxy.
        e["construction_score"] = float(1.0 if cone.rel in ("on_path", "near_path") else 0.0)
        # Surface risk is not directly observable w/o a model; use (1-visibility)*0.5 as weak proxy.
        e["surface_risk"] = float(max(0.0, min(1.0, (1.0 - e["visibility_score"]) * 0.5)))

        return e

    @staticmethod
    def _nav_cmd_from_onehot(driving_command: Optional[Sequence[int]]) -> str:
        cmds = ["LEFT", "STRAIGHT", "RIGHT"]
        if driving_command is None:
            return "UNKNOWN"
        for i, v in enumerate(driving_command):
            try:
                if int(v) == 1:
                    return cmds[i]
            except Exception:
                continue
        return "UNKNOWN"
