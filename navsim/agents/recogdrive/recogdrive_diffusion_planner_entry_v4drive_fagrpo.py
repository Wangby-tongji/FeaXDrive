# Copyright 2025 The Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import lzma
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from timm.models.layers import Mlp
from torch import nn
from torch.distributions import Beta, Normal, kl_divergence
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score_fagrpo import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_fagrpo import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from .blocks.encoder import (
    ActionEncoder,
    SinusoidalPositionalEncoding,
    StateAttentionEncoder,
    SwiGLUFFN,
)
from .recogdrive_dit import LightningDiT
from .trajectory_projector_v4drive import ProjectionLimits, ProxConfig, TrajectoryProjector

@dataclass
class FlowConfig:
    """Configuration specific to Flow Matching."""
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    mean_variance_net: bool = False

@dataclass
class DDPMConfig:
    """Configuration specific to DDPM."""
    num_train_timesteps: int = 100

@dataclass
class DDIMConfig:
    """Configuration specific to DDIM."""
    num_train_timesteps: int = 100
    ddim_eta: float = 0.0

@dataclass
class GRPOConfig:
    """Configuration specific to GRPO training."""
    denoised_clip_value: float = 1.0
    eval_randn_clip_value: float = 1.0
    randn_clip_value: float = 5.0
    final_action_clip_value: float = 1.0
    eps_clip_value: Optional[float] = None
    eval_min_sampling_denoising_std: float = 0.0001
    min_sampling_denoising_std: float = 0.04
    min_logprob_denoising_std: float = 0.1
    clip_advantage_lower_quantile: float = 0.0
    clip_advantage_upper_quantile: float = 1.0
    gamma_denoising: float = 0.6
    
    metric_cache_path: str = ""
    reference_policy_checkpoint: str = ""
    scorer_config: PDMScorerConfig = field(default_factory=lambda: PDMScorerConfig(
        progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0
    ))


@dataclass
class ReCogDriveDiffusionPlannerConfig(PretrainedConfig):
    """A refined configuration for the ReCogDriveDiffusionPlanner."""
    # --- Core Architecture ---
    diffusion_model_cfg: dict = field(default_factory=dict)
    input_embedding_dim: int = 1536
    hidden_size: int = 1024
    action_dim: int = 3
    action_horizon: int = 8
    add_pos_embed: bool = True
    max_seq_len: int = 8
    ego_status_encoder_type: Literal['mlp', 'attention'] = 'mlp'

    sampling_method: Literal['flow', 'ddpm', 'ddim'] = 'ddim'

    # What the diffusion model predicts for DDPM/DDIM training & sampling.
    # - 'eps': predict noise (original behavior)
    # - 'x0' : predict clean actions directly (JiT-style denoising)
    pred_type: Literal['eps', 'x0'] = 'eps'

    # ---- Dynamics feasibility projection (used in sampling and/or training) ----
    use_constraint_projection: bool = False
    # Projection (feasibility) in diffusion sampling.
    # In the long-tail setting here, baseline already satisfies v/a/j, but may violate
    # curvature / lateral-acc constraints; so projection focuses on kappa / a_lat.
    proj_dt: float = 0.5
    # Fixed curvature limit (legacy). Prefer proj_use_kappa_adapt + proj_kappa_geo_max.
    proj_kappa_max: Optional[float] = None      # 1/m
    # Vehicle kinematic/geometric curvature limit (1/m), e.g. 0.166.
    proj_kappa_geo_max: Optional[float] = None  # 1/m
    # If True, use speed-adaptive curvature bound:
    #   kappa_max_adapt(t) = min(proj_kappa_geo_max, proj_a_lat_max / (v(t)^2 + eps))
    proj_use_kappa_adapt: bool = False
    proj_a_lat_max: Optional[float] = None      # m/s^2
    proj_smooth_ksize: int = 5                  # odd kernel for smoothing (scheme B)
    proj_step_size: float = 0.1                # proximal GD step size
    proj_lambda_kappa: float = 1.0             # weight in proximal objective
    proj_lambda_a_lat: float = 1.0             # weight in proximal objective
    proj_iters: int = 1                        # number of unrolled proximal steps
    proj_keep_start: bool = True

    # ---- Constraint-consistency training (lightweight regularization) ----
    # Training-time constraint consistency (optional)
    lambda_dyn: float = 0.0              # soft violation penalty weight (kappa/a_lat)
    lambda_proj: float = 0.0             # ||x0 - Pi(x0)||^2 weight
    lambda_proj_supervise: float = 0.0   # ||Pi(x0) - x_gt||^2 weight
    proj_stop_grad: bool = True          # detach Pi(x0) target for stability

    # Logging / analysis: record ΔΠ (projection correction) curve across sampling steps.
    record_delta_pi_curve: bool = False
    num_inference_steps: int = 5
    model_dtype: str = "float16"
    grpo: bool = False
    vlm_size: str = 'large'

    # ---- Drivable-area guidance (inference-time x0-guidance) ----
    # Enable guidance only if an SDF context is provided at runtime.
    use_drivable_guidance: bool = False
    # Apply guidance only on the last K denoising steps.
    drivable_guidance_last_k: int = 3
    # Gradient step size in *normalized action space* (x0 in [-1,1]).
    drivable_guidance_step_size: float = 0.05
    # Soft barrier parameters in meters: penalize if SDF < margin.
    drivable_guidance_margin_m: float = 0.30
    drivable_guidance_beta_m: float = 0.20
    # If a query point is outside the SDF ROI, treat it as strongly off-road.
    drivable_guidance_oob_sdf_m: float = -10.0
    # Schedule: step_scale = (progress ** power), where progress in (0,1].
    drivable_guidance_power: float = 1.0
    # Heading update scale in drivable guidance. 0 disables heading correction (legacy behavior).
    drivable_guidance_heading_scale: float = 0.1

    # ---- Off-road gating for drivable guidance ----
    # If True, apply drivable guidance only when the current x0 trajectory is off-road
    # (min SDF over footprint corners < -drivable_guidance_offroad_eps_m).
    drivable_guidance_only_offroad: bool = False
    # Tolerance (meters) to absorb raster/SDF quantization; treat SDF >= -eps as on-road.
    drivable_guidance_offroad_eps_m: float = 0.0
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    
    flow_cfg: FlowConfig = field(default_factory=FlowConfig)
    ddpm_cfg: DDPMConfig = field(default_factory=DDPMConfig)
    ddim_cfg: DDIMConfig = field(default_factory=DDIMConfig)
    grpo_cfg: GRPOConfig = field(default_factory=GRPOConfig)


class ReCogDriveDiffusionPlanner(nn.Module):
    config_class = ReCogDriveDiffusionPlannerConfig

    def __init__(self, config: ReCogDriveDiffusionPlannerConfig):
        super().__init__()
        self.config = config
        
        self.model = LightningDiT(**config.diffusion_model_cfg)

        self.his_traj_encoder = Mlp(
            in_features=12,
            hidden_features=config.hidden_size,
            out_features=config.input_embedding_dim,
            norm_layer=nn.LayerNorm
        )

        if config.ego_status_encoder_type == 'attention':
            self.ego_status_encoder = StateAttentionEncoder(
                state_dim=8,
                embed_dim=config.input_embedding_dim,
                num_kinematic_states=4 
            )
        else: 
            self.ego_status_encoder = Mlp(
                in_features=8,
                hidden_features=config.hidden_size,
                out_features=config.input_embedding_dim,
                norm_layer=nn.LayerNorm
            )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=config.input_embedding_dim,
        )
        if config.vlm_size == "large":
            self.feature_encoder = nn.Linear(3584, config.input_embedding_dim)
        else:
            self.feature_encoder = nn.Linear(1536, config.input_embedding_dim)
            
        self.fusion_projector = nn.Linear(config.input_embedding_dim * 3, config.input_embedding_dim)

        output_dim = 2 * config.action_dim if (
            config.sampling_method == 'flow' and config.flow_cfg.mean_variance_net
        ) else config.action_dim
        
        self.action_decoder = Mlp(
            in_features=self.model.output_dim,
            hidden_features=config.hidden_size,
            out_features=output_dim,
            norm_layer=nn.LayerNorm
        )
        
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        if self.config.sampling_method == 'flow':
            self._init_flow_sampler(config.flow_cfg)
        elif self.config.sampling_method == 'ddpm':
            self._init_ddpm_sampler(config.ddpm_cfg)
        elif self.config.sampling_method == 'ddim':
            self._init_ddim_sampler(config.ddim_cfg)

        # Feasibility projector (optional):
        # - Sampling: step-wise projection participates in the reverse update.
        # - Training: provides constraint-consistency regularization.
        self.projector: Optional[TrajectoryProjector] = None
        if (
            config.use_constraint_projection
            or config.lambda_dyn > 0
            or config.lambda_proj > 0
            or config.lambda_proj_supervise > 0
        ):
            limits = ProjectionLimits(
                dt=float(config.proj_dt),
                kappa_max=config.proj_kappa_max,
                kappa_geo_max=config.proj_kappa_geo_max,
                a_lat_max=config.proj_a_lat_max,
                use_kappa_adapt=bool(config.proj_use_kappa_adapt),
            )
            prox = ProxConfig(
                steps=int(config.proj_iters),
                step_size=float(config.proj_step_size),
                lambda_kappa=float(config.proj_lambda_kappa),
                lambda_a_lat=float(config.proj_lambda_a_lat),
                smooth_ksize=int(config.proj_smooth_ksize),
                keep_start=bool(config.proj_keep_start),
            )
            self.projector = TrajectoryProjector(limits=limits, prox=prox)

        if config.grpo:
            self._init_grpo(config.grpo_cfg)

        # Vehicle parameters for footprint corners (rear-axle frame).
        self._veh = get_pacifica_parameters()

        # Optional runtime drivable SDF context (set per-token during evaluation).
        # Expected keys: valid, sdf[1,1,H,W], x_min,x_max,y_min,y_max,res,H,W.
        self._drivable_ctx: Optional[Dict[str, object]] = None

    # -------------------------
    # Drivable guidance plumbing
    # -------------------------
    def set_drivable_sdf_context(self, ctx: Optional[Dict[str, object]]) -> None:
        """Attach per-scenario drivable SDF context for inference-time guidance."""
        self._drivable_ctx = ctx

    def _corners_xy(self, poses: torch.Tensor) -> torch.Tensor:
        """Compute 4 footprint corners in ego-frame from poses.

        Args:
            poses: [B,T,3] (x,y,heading) in ego-frame, rear-axle reference.
        Returns:
            corners: [B,T,4,2] in ego-frame.
        """
        x, y, th = poses[..., 0], poses[..., 1], poses[..., 2]
        c, s = torch.cos(th), torch.sin(th)
        front = float(self._veh.front_length)
        rear = float(self._veh.rear_length)
        hw = float(self._veh.width) * 0.5

        # Unit vectors along heading and left-normal.
        ex_x, ex_y = c, s
        ey_x, ey_y = -s, c

        # Corner offsets from rear axle.
        fx = front * ex_x
        fy = front * ex_y
        rx = -rear * ex_x
        ry = -rear * ex_y
        lx = hw * ey_x
        ly = hw * ey_y

        # FL, FR, RL, RR
        fl = torch.stack([x + fx + lx, y + fy + ly], dim=-1)
        fr = torch.stack([x + fx - lx, y + fy - ly], dim=-1)
        rl = torch.stack([x + rx + lx, y + ry + ly], dim=-1)
        rr = torch.stack([x + rx - lx, y + ry - ly], dim=-1)
        return torch.stack([fl, fr, rl, rr], dim=-2)

    def _sample_sdf(self, pts_xy: torch.Tensor, ctx: Dict[str, object]) -> torch.Tensor:
        """Sample SDF at pts using bilinear grid_sample.

        Args:
            pts_xy: [B,N,2] in ego-frame (meters).
            ctx: drivable context dict.
        Returns:
            sdf: [B,N] in meters (positive inside, negative outside).
        """
        sdf_grid: torch.Tensor = ctx["sdf"]  # [1,1,H,W] or [B,1,H,W]
        x_min = float(ctx["x_min"]) ; x_max = float(ctx["x_max"])
        y_min = float(ctx["y_min"]) ; y_max = float(ctx["y_max"])
        res = float(ctx["res"]) ; H = int(ctx["H"]) ; W = int(ctx["W"])

        x = pts_xy[..., 0]
        y = pts_xy[..., 1]

        # Out-of-bounds mask (explicit, to avoid padding artifacts).
        oob = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)

        col = (x - x_min) / res
        row = (y_max - y) / res

        gx = 2.0 * (col / max(W - 1, 1)) - 1.0
        gy = 2.0 * (row / max(H - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1)  # [B,N,2]
        grid = grid.unsqueeze(2)  # [B,N,1,2]

        # Expand sdf grid to batch if needed.
        if sdf_grid.shape[0] == 1 and pts_xy.shape[0] > 1:
            sdf_in = sdf_grid.expand(pts_xy.shape[0], -1, -1, -1)
        else:
            sdf_in = sdf_grid

        import torch.nn.functional as _F
        sdf = _F.grid_sample(
            sdf_in,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sdf = sdf.squeeze(1).squeeze(-1)  # [B,N]

        if torch.any(oob):
            sdf = torch.where(
                oob,
                torch.full_like(sdf, float(getattr(self.config, "drivable_guidance_oob_sdf_m", -10.0))),
                sdf,
            )
        return sdf

    def _apply_drivable_guidance(self, x0_norm: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Inference-time x0 guidance to keep footprint inside drivable area."""
        if not bool(getattr(self.config, "use_drivable_guidance", False)):
            return x0_norm
        ctx = getattr(self, "_drivable_ctx", None)
        if not ctx or not bool(ctx.get("valid", False)):
            return x0_norm

        # Apply only on last K steps.
        K_total = int(getattr(self.config, "num_inference_steps", 0) or getattr(self, "ddim_steps", 0) or 0)
        last_k = int(max(0, getattr(self.config, "drivable_guidance_last_k", 0)))
        if K_total <= 0 or last_k <= 0:
            return x0_norm

        i = int(index[0].item())
        if i < (K_total - last_k):
            return x0_norm

        progress = float(i - (K_total - last_k) + 1) / float(last_k)  # (0,1]
        power = float(getattr(self.config, "drivable_guidance_power", 1.0))
        step = float(getattr(self.config, "drivable_guidance_step_size", 0.05)) * (progress ** power)
        margin_m = float(getattr(self.config, "drivable_guidance_margin_m", 0.30))
        beta_m = float(getattr(self.config, "drivable_guidance_beta_m", 0.20))

        only_offroad = bool(getattr(self.config, "drivable_guidance_only_offroad", False))
        eps_m = float(getattr(self.config, "drivable_guidance_offroad_eps_m", 0.0))

        x0_in = x0_norm
        x0_fp32 = x0_in.detach().to(torch.float32)



        # Trajectory-level gating: keep the sampling trigger logic (last_k steps) unchanged,
        # but skip guidance entirely when the current x0 trajectory is already on-road.
        # NOTE: Use per-sample gating (important when batch>1).
        need_guidance = None
        if only_offroad:
            with torch.no_grad():
                poses0 = self.denorm_odo(x0_fp32)  # [B,T,3]
                corners0 = self._corners_xy(poses0)  # [B,T,4,2]
                pts0 = corners0.reshape(corners0.shape[0], -1, 2)  # [B,T*4,2]
                sdf0 = self._sample_sdf(pts0, ctx)  # [B,T*4]
                min_sdf0 = sdf0.min(dim=1).values  # [B]
                need_guidance = min_sdf0 < (-eps_m)
                if not bool(need_guidance.any().item()):
                    return x0_norm

        with torch.enable_grad():
            x0_g = x0_fp32.requires_grad_(True)
            poses = self.denorm_odo(x0_g)  # [B,T,3]
            corners = self._corners_xy(poses)  # [B,T,4,2]
            pts = corners.reshape(corners.shape[0], -1, 2)
            sdf = self._sample_sdf(pts, ctx)

            import torch.nn.functional as _F

            if only_offroad:

                # Off-road gating enabled. Apply guidance only to samples that violate:
                #   min_sdf0 < -eps_m
                if need_guidance is None:
                    # Fallback: treat all samples as needing guidance.
                    need_guidance = torch.ones(sdf.shape[0], device=sdf.device, dtype=torch.bool)

                trigger = need_guidance.to(dtype=sdf.dtype).unsqueeze(1)  # [B,1]

                # Push footprint points to a SAFE margin inside drivable area:
                #   enforce sdf >= margin_m (treat margin_m as margin_safe here).
                viol = (margin_m - sdf)  # >0 when sdf < margin_safe
                mask = (viol > 0).to(dtype=viol.dtype) * trigger  # [B,N]

                barrier = _F.softplus(viol / max(beta_m, 1e-6)) * mask  # [B,N]

                denom = mask.sum(dim=1).clamp(min=1.0)  # [B]
                loss_per = barrier.sum(dim=1) / denom   # [B]

                # Average loss over the triggered subset (keep gradients zero for non-triggered).
                loss = loss_per.sum() / trigger.sum().clamp(min=1.0)
            else:

                # Original behavior: apply near-boundary shaping to all trajectories.

                barrier = _F.softplus((margin_m - sdf) / max(beta_m, 1e-6))

                loss = barrier.mean()
            grad = torch.autograd.grad(loss, x0_g, retain_graph=False, create_graph=False)[0]

        grad = grad.detach()

        # --- Heading-aware guidance (fix issue #4) ---
        # Previous behavior zeroed heading gradients, which can prevent correcting corner-based off-road
        # violations that are best fixed by a small yaw adjustment. We allow a scaled heading update.
        heading_scale = float(getattr(self.config, "drivable_guidance_heading_scale", 0.1))

        # Normalize XY update per-sample to keep a stable step size.
        grad_xy = grad[..., :2]
        gxy = grad_xy.reshape(grad.shape[0], -1)
        xy_norm = torch.linalg.norm(gxy, dim=-1, keepdim=True).clamp(min=1e-6)
        grad_xy = grad_xy / xy_norm.view(-1, 1, 1)

        # Heading update uses sign-normalized gradient, scaled by heading_scale.
        if heading_scale <= 0.0:
            grad_h = torch.zeros_like(grad[..., 2:3])
        else:
            grad_h_raw = grad[..., 2:3]
            h_norm = grad_h_raw.abs().clamp(min=1e-6)
            grad_h = heading_scale * (grad_h_raw / h_norm)

        grad = torch.cat([grad_xy, grad_h], dim=-1)

        x0_new = x0_fp32 - step * grad
        return x0_new.to(dtype=x0_in.dtype)

    def _should_project(self) -> bool:
        return bool(getattr(self.config, 'use_constraint_projection', False) and self.projector is not None)

    def _project_x0_norm(self, x0_norm: torch.Tensor) -> torch.Tensor:
        """Project a normalized x0 trajectory in-place via metric-space projection."""
        if self.projector is None:
            return x0_norm
        x0_denorm = self.denorm_odo(x0_norm)
        x0_denorm_proj = self.projector.project(x0_denorm)
        x0_norm_proj = self.norm_odo(x0_denorm_proj)
        return x0_norm_proj

    def _maybe_record_delta_pi(self, x0_pre_norm: torch.Tensor, x0_post_norm: torch.Tensor) -> None:
        """Record ΔΠ = ||x0 - Π(x0)|| for analysis.

        We record the mean XY correction per diffusion sampling step. The curve is stored
        in `self._delta_pi_curve`, which is (re)initialized at the beginning of sampling.
        """
        if not bool(getattr(self.config, "record_delta_pi_curve", False)):
            return
        if not hasattr(self, "_delta_pi_curve"):
            return
        try:
            x0_pre_m = self.denorm_odo(x0_pre_norm)
            x0_post_m = self.denorm_odo(x0_post_norm)
            delta = torch.linalg.norm(x0_pre_m[..., :2] - x0_post_m[..., :2], dim=-1)  # [B,H]
            self._delta_pi_curve.append(delta.mean().detach())
        except Exception:
            # Never break planning due to logging.
            return

    def _init_flow_sampler(self, cfg: FlowConfig):
        """Initializes components required for Flow Matching."""
        self.beta_dist = Beta(cfg.noise_beta_alpha, cfg.noise_beta_beta)
        self.num_timestep_buckets = cfg.num_timestep_buckets

    def _init_ddpm_sampler(self, cfg: DDPMConfig):
        """Initializes buffers required for DDPM, using original naming."""
        ddpm_betas = self.cosine_beta_schedule(cfg.num_train_timesteps)
        self.register_buffer('ddpm_betas', ddpm_betas)

        ddpm_alphas = 1.0 - ddpm_betas
        self.register_buffer('ddpm_alphas', ddpm_alphas)

        ddpm_alphas_cumprod = torch.cumprod(ddpm_alphas, dim=0)
        self.register_buffer('ddpm_alphas_cumprod', ddpm_alphas_cumprod)

        ddpm_alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), ddpm_alphas_cumprod[:-1]])
        self.register_buffer('ddpm_alphas_cumprod_prev', ddpm_alphas_cumprod_prev)

        self.register_buffer('ddpm_sqrt_alphas_cumprod', torch.sqrt(ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod - 1.0))

        ddpm_var = ddpm_betas * (1.0 - ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod)
        self.register_buffer('ddpm_var', ddpm_var)
        self.register_buffer('ddpm_logvar_clipped', torch.log(ddpm_var.clamp(min=1e-20)))
        
        self.register_buffer('ddpm_mu_coef1', ddpm_betas * torch.sqrt(ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_mu_coef2', (1.0 - ddpm_alphas_cumprod_prev) * torch.sqrt(ddpm_alphas) / (1.0 - ddpm_alphas_cumprod))

    def _init_ddim_sampler(self, cfg: DDIMConfig):
        """Initializes buffers required for DDIM sampling, using original naming."""
        self._init_ddpm_sampler(DDPMConfig(num_train_timesteps=cfg.num_train_timesteps))

        self.eta = EtaFixed(base_eta=1.0).to(self.device)
        for param in self.eta.parameters():
            param.requires_grad = False
        ddim_steps = self.config.num_inference_steps
        self.ddim_steps = ddim_steps
        self.ft_denoising_steps = ddim_steps
        ddim_eta = cfg.ddim_eta 

        self.ddpm_num_train_timesteps = cfg.num_train_timesteps
        step_ratio = self.ddpm_num_train_timesteps // ddim_steps
        ddim_t = torch.arange(0, ddim_steps) * step_ratio
        self.register_buffer('ddim_t_schedule', ddim_t.long()) # Die originalen Zeitpunkte

        ddim_alphas = self.ddpm_alphas_cumprod[self.ddim_t_schedule].clone().to(torch.float32)
        ddim_alphas_prev = torch.cat([
            torch.tensor([1.0], dtype=torch.float32),
            self.ddpm_alphas_cumprod[self.ddim_t_schedule[:-1]]
        ])
        ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas) ** 0.5

        ddim_sigmas = ddim_eta * (
            (1 - ddim_alphas_prev) / (1 - ddim_alphas) *
            (1 - ddim_alphas / ddim_alphas_prev)
        )**0.5

        # Flip all for sampling order (T -> 0)
        def flip_buffer(name, tensor):
            self.register_buffer(name, torch.flip(tensor, [0]))

        flip_buffer('ddim_t', self.ddim_t_schedule)
        flip_buffer('ddim_alphas', ddim_alphas)
        flip_buffer('ddim_alphas_sqrt', torch.sqrt(ddim_alphas))
        flip_buffer('ddim_alphas_prev', ddim_alphas_prev)
        flip_buffer('ddim_sqrt_one_minus_alphas', ddim_sqrt_one_minus_alphas)
        flip_buffer('ddim_sigmas', ddim_sigmas)

    def _init_grpo(self, cfg: GRPOConfig):
        """Initializes components and hyperparameters for GRPO training."""
        self.denoised_clip_value = cfg.denoised_clip_value
        self.eval_randn_clip_value = cfg.eval_randn_clip_value
        self.randn_clip_value = cfg.randn_clip_value
        self.final_action_clip_value = cfg.final_action_clip_value
        self.eps_clip_value = cfg.eps_clip_value
        self.eval_min_sampling_denoising_std = cfg.eval_min_sampling_denoising_std
        self.min_sampling_denoising_std = cfg.min_sampling_denoising_std
        self.min_logprob_denoising_std = cfg.min_logprob_denoising_std
        self.clip_advantage_lower_quantile = cfg.clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = cfg.clip_advantage_upper_quantile
        self.gamma_denoising = cfg.gamma_denoising
        
        self.metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        self.train_scorer = PDMScorer(proposal_sampling, cfg.scorer_config)
        
        try:
            state_dict = torch.load(cfg.reference_policy_checkpoint, map_location="cpu")["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in state_dict.items():
                if k.startswith("agent.action_head."):
                    k2 = k[len("agent.action_head."):]
                else:
                    k2 = k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
                else:
                    print(f"Skip loading '{k}' → '{k2}' (checkpoint shape {tuple(v.shape)} vs model shape {tuple(model_dict.get(k2, v).shape)})")
            self.load_state_dict(filtered_ckpt, strict=True)
        except FileNotFoundError:
            print(f"Warning: GRPO checkpoint not found at {cfg.reference_policy_checkpoint}. Skipping loading.")
        
        self.old_policy = copy.deepcopy(self)
        self.old_policy.eval()
        for param in self.old_policy.parameters():
            param.requires_grad = False

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Calculates a cosine noise schedule as proposed in the iDDPM paper.
        
        This method is static as it does not depend on the instance's state.
        """
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        return torch.tensor(betas_clipped, dtype=dtype)

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extracts values from tensor `a` at indices `t` and reshapes them
        to be broadcastable with a tensor of shape `x_shape`.
        
        This method is static as it does not depend on the instance's state.
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def make_timesteps(batch_size: int, i: int, device: torch.device) -> torch.Tensor:
        """
        Creates a tensor of a constant value `i` for a given batch size and device.
        
        This method is static as it does not depend on the instance's state.
        """
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        return t


    def set_frozen_modules_to_eval_mode(self):
        """
        Sets frozen parts of the model to evaluation mode during training.
        This is necessary to disable behaviors like dropout in the frozen layers.
        """
        if self.training:

            if not self.config.tune_projector:
                self.his_traj_encoder.eval()
                self.ego_status_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.feature_encoder.eval()
                self.fusion_projector.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            
            if not self.config.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """Samples time for training based on the sampling method."""
        if self.config.sampling_method == 'flow':
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            return (self.config.flow_cfg.noise_s - sample) / self.config.flow_cfg.noise_s
        elif self.config.sampling_method in ['ddpm', 'ddim']:
            return torch.randint(0, self.ddpm_num_train_timesteps, (batch_size,), device=device).long()
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")


    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        deterministic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and log variance of the reverse process p(x_{t-1} | x_t).
        Also returns the predicted x0.
        """
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)
        action_features = self.action_encoder(x, t)
        if hasattr(self, 'position_embedding'):
            pos_ids = torch.arange(action_features.shape[1], device=x.device)
            action_features = action_features + self.position_embedding(pos_ids)

        vl_features_mean = vl_features.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
        fused_input = self.fusion_projector(
            torch.cat((his_traj_features, vl_features_mean, action_features), dim=2)
        )

        model_output = self.model(
            hidden_states=fused_input,
            encoder_hidden_states=vl_features,
            conditioning_features=ego_status_features,
            timesteps=t
        )

        # Model output semantics for DDPM/DDIM:
        #   - pred_type == 'eps': model predicts noise (eps)
        #   - pred_type == 'x0' : model predicts clean actions x0 directly
        model_pred = self.action_decoder(model_output)

        if self.config.sampling_method == 'ddpm':
            if self.config.pred_type == 'eps':
                pred_noise = model_pred
                x_recon = self.extract(self.ddpm_sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                          self.extract(self.ddpm_sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
            elif self.config.pred_type == 'x0':
                # Directly predict x0 (clean actions)
                x_recon = model_pred
            else:
                raise ValueError(f"Unsupported pred_type: {self.config.pred_type}")

        elif self.config.sampling_method == 'ddim':
            alpha_t = self.extract(self.ddim_alphas, index, x.shape)
            sqrt_one_minus_alpha_t = self.extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)

            if self.config.pred_type == 'eps':
                pred_noise = model_pred
                x_recon = (x - sqrt_one_minus_alpha_t * pred_noise) / (alpha_t**0.5)
            elif self.config.pred_type == 'x0':
                x_recon = model_pred
            else:
                raise ValueError(f"Unsupported pred_type: {self.config.pred_type}")

        else:
            raise ValueError(f"p_mean_variance not supported for method: {self.config.sampling_method}")

        denoised_clip_value = getattr(self, 'denoised_clip_value', 1.0)
        x_recon.clamp_(-denoised_clip_value, denoised_clip_value)

        # Drivable-area x0-guidance (optional): soft push back into drivable region.
        # Inserted before the kinematic projection so feasibility projection can
        # refine the guided result.
        x_recon = self._apply_drivable_guidance(x_recon, index)
        x_recon = x_recon.clamp(-denoised_clip_value, denoised_clip_value)

        # Step-wise feasibility projection: project x0 (in metric space) and
        # use the projected x0 to compute eps and the reverse update.
        if self._should_project():
            x_recon_pre = x_recon
            x_recon = self._project_x0_norm(x_recon)
            self._maybe_record_delta_pi(x_recon_pre, x_recon)
            x_recon = x_recon.clamp(-denoised_clip_value, denoised_clip_value)

        # ------------------------------------------------------------
        # Step-wise feasibility projection (key point):
        #   - Project x0 (denoised prediction) onto the feasible set.
        #   - Use the projected x0 to back out eps and update x_{t-1}.
        # This ensures projection *participates in the reverse chain*, not
        # merely as a final post-processing step.
        # ------------------------------------------------------------
        # if self.projector is not None and self.config.use_constraint_projection:
        #     # Convert to metric space -> project -> convert back to normalized action space.
        #     x_recon_metric = self.denorm_odo(x_recon)
        #     x_recon_metric = self.projector.project(x_recon_metric)
        #     x_recon = self.norm_odo(x_recon_metric)
        #     x_recon.clamp_(-denoised_clip_value, denoised_clip_value)

        if self.config.sampling_method == 'ddpm':
            model_mean = self.extract(self.ddpm_mu_coef1, t, x.shape) * x_recon + \
                         self.extract(self.ddpm_mu_coef2, t, x.shape) * x
            model_log_variance = self.extract(self.ddpm_logvar_clipped, t, x.shape)
        elif self.config.sampling_method == 'ddim':
            alpha_prev = self.extract(self.ddim_alphas_prev, index, x.shape)
            
            pred_noise = (x - (alpha_t**0.5) * x_recon) / sqrt_one_minus_alpha_t

            eps_clip_value = getattr(self, 'eps_clip_value', None)
            if eps_clip_value is not None:
                pred_noise.clamp_(-eps_clip_value, eps_clip_value)

            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(x).unsqueeze(1)

            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)

            pred_dir_xt = (1.0 - alpha_prev - sigma**2).clamp(min=0).sqrt() * pred_noise
            model_mean = (alpha_prev**0.5) * x_recon + pred_dir_xt
            model_log_variance = torch.log(sigma**2 + 1e-20)

        return model_mean, model_log_variance, x_recon

    def forward(self, vl_features: torch.Tensor, action_input: BatchFeature) -> BatchFeature:
        """
        Computes the training loss for a given batch.

        Args:
            vl_features (torch.Tensor): The vision-language features from the backbone.
            action_input (BatchFeature): A batch containing ground truth actions and other conditioning.

        Returns:
            BatchFeature: A batch containing the computed loss.
        """
        
        vl_embeds = self.feature_encoder(vl_features)
        his_traj_features = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)
        ego_status_features = self.ego_status_encoder(action_input.status_feature)
        
        gt_actions = self.norm_odo(action_input.action)

        if self.config.sampling_method == 'flow':
            noise = torch.randn_like(gt_actions)
            t_cont = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            t_cont_reshaped = t_cont[:, None, None]
            
            noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
            velocity_target = gt_actions - noise
            t_discrete = (t_cont * self.num_timestep_buckets).long()

            action_features = self.action_encoder(noisy_actions, t_discrete)
            if hasattr(self, 'position_embedding'):
                pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
                action_features += self.position_embedding(pos_ids)
            
            vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            fused_input = self.fusion_projector(
                torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2)
            )

            model_output = self.model(fused_input, vl_embeds, ego_status_features, t_discrete)
            pred_velocity = self.action_decoder(model_output)
            loss = F.mse_loss(pred_velocity, velocity_target, reduction='mean')
            return BatchFeature(data={"loss": loss, "loss_base": loss.detach()})
        else: 
            noise = torch.randn_like(gt_actions)
            t_discrete = self.sample_time(gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype)
            
            noisy_actions = (
                self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, gt_actions.shape) * gt_actions +
                self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, gt_actions.shape) * noise
            )
            
            action_features = self.action_encoder(noisy_actions, t_discrete)
            if hasattr(self, 'position_embedding'):
                pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
                action_features += self.position_embedding(pos_ids)
            
            vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            fused_input = self.fusion_projector(
                torch.cat((his_traj_features, vl_embeds_mean, action_features), dim=2)
            )
            
            model_output = self.model(fused_input, vl_embeds, ego_status_features, t_discrete)
            model_pred = self.action_decoder(model_output)

            # ------------------------------------------------------------
            # Base diffusion objective (eps-loss or x0-loss)
            # Additionally derive x0_pred (normalized) for constraint-consistency
            # regularization, regardless of pred_type.
            # ------------------------------------------------------------
            if self.config.pred_type == 'eps':
                pred_noise = model_pred
                base_loss = F.mse_loss(pred_noise, noise, reduction='mean')
                sqrt_alpha = self.extract(self.ddpm_sqrt_alphas_cumprod, t_discrete, gt_actions.shape)
                sqrt_one_minus_alpha = self.extract(self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, gt_actions.shape)
                x0_pred = (noisy_actions - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha.clamp(min=1e-8)
            elif self.config.pred_type == 'x0':
                x0_pred = model_pred
                base_loss = F.mse_loss(x0_pred, gt_actions, reduction='mean')
            else:
                raise ValueError(f"Unsupported pred_type: {self.config.pred_type}")

            denoised_clip_value = getattr(self, 'denoised_clip_value', 1.0)
            x0_pred = x0_pred.clamp(-denoised_clip_value, denoised_clip_value)

            total_loss = base_loss
            out_dict = {
                "loss": total_loss,
                "loss_base": base_loss.detach(),
            }

            # ------------------------------------------------------------
            # Constraint-consistency training:
            #  (A) Soft dynamics violation penalty
            #  (B) Projection consistency: ||x0 - Pi(x0)||^2 (optionally stop-grad)
            #  (C) Projection supervised: ||Pi(x0) - x_gt||^2 (small weight)
            # All penalties are computed in metric space for meaningful thresholds.
            # ------------------------------------------------------------
            if self.projector is not None and (
                self.config.lambda_dyn > 0 or self.config.lambda_proj > 0 or self.config.lambda_proj_supervise > 0
            ):
                x0_metric = self.denorm_odo(x0_pred)

                if self.config.lambda_dyn > 0:
                    dyn_pen, dyn_stats = self.projector.violation(x0_metric, reduction="mean")
                    total_loss = total_loss + float(self.config.lambda_dyn) * dyn_pen
                    out_dict["loss_dyn"] = dyn_pen.detach()
                    for k, v in dyn_stats.items():
                        out_dict[f"dyn_{k}"] = v.detach()
                
                x0_proj_metric = None

                need_proj = (self.config.lambda_proj > 0) or (self.config.lambda_proj_supervise > 0)
                if need_proj:
                    x0_proj_metric = self.projector.project(x0_metric)  # 只投影一次

                # 1) projection consistency: ||x0 - Pi(x0)||^2
                if self.config.lambda_proj > 0:
                    x0_proj_for_proj = x0_proj_metric.detach() if bool(self.config.proj_stop_grad) else x0_proj_metric
                    proj_loss = F.mse_loss(x0_metric[..., :2], x0_proj_for_proj[..., :2], reduction="mean")
                    total_loss = total_loss + float(self.config.lambda_proj) * proj_loss
                    out_dict["loss_proj"] = proj_loss.detach()

                # 2) projection-supervised: ||Pi(x0) - x_gt||^2
                if self.config.lambda_proj_supervise > 0:
                    x0_proj_for_pi = x0_proj_metric.detach() if bool(self.config.proj_stop_grad) else x0_proj_metric
                    pi_loss = F.mse_loss(x0_proj_for_pi[..., :2], action_input.action[..., :2], reduction="mean")
                    total_loss = total_loss + float(self.config.lambda_proj_supervise) * pi_loss
                    out_dict["loss_pi"] = pi_loss.detach()

            out_dict["loss"] = total_loss

        return BatchFeature(data=out_dict)

    def get_action(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> BatchFeature:
        """
        Generates action trajectories via the configured sampling method.

        This method strictly preserves the original logic for each sampler,
        including specific clipping and noise handling for DDPM and DDIM.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            action_input (BatchFeature): Input containing conditioning features like
                historical trajectory and ego status.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start
                the denoising from. If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic (eta=0).

        Returns:
            BatchFeature: A batch containing the final predicted trajectory.
        """
        vl_embeds = self.feature_encoder(vl_features)

        # (Re)initialize ΔΠ curve container if requested.
        if bool(getattr(self.config, "record_delta_pi_curve", False)):
            self._delta_pi_curve = []
        
        history_embeds = self.his_traj_encoder(
            action_input.his_traj.unsqueeze(1)
        ).repeat(1, self.config.action_horizon, 1)

        ego_embeds = self.ego_status_encoder(
            action_input.status_feature
        )

        B, D = vl_embeds.shape[0], self.config.action_dim
        device, dtype = vl_embeds.device, vl_embeds.dtype
        
        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t = torch.full((B,), idx, device=device, dtype=torch.long)

                action_features = self.action_encoder(current_actions, t)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((history_embeds, vl_embeds_mean, action_features), dim=2)
                )
                
                model_output = self.model(fused_input, vl_embeds, ego_embeds, t)
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow

        elif self.config.sampling_method == 'ddpm':
            step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
            timesteps_to_iterate = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            
            for i, t_int in enumerate(timesteps_to_iterate):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, vl_embeds, history_embeds, ego_embeds, deterministic
                )

                noise_sample = torch.randn_like(current_actions)
                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                if t_int == 0:
                    std.zero_()
                else:
                    std = torch.clamp(std, min=1e-3)

                if hasattr(self, 'eval_randn_clip_value') and self.eval_randn_clip_value is not None:
                    noise_sample.clamp_(-self.eval_randn_clip_value, self.eval_randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None and i == len(timesteps_to_iterate) - 1:
                    current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)

        elif self.config.sampling_method == 'ddim':
            eval_min_sampling_denoising_std = getattr(self, 'eval_min_sampling_denoising_std', 0.0001)
            eval_randn_clip_value = getattr(self, 'eval_randn_clip_value', 1.0)
            for i in range(self.ddim_steps):
                t_batch = self.make_timesteps(B, self.ddim_t[i], device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, vl_embeds, history_embeds, ego_embeds, deterministic
                )

                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                noise_sample = torch.randn_like(current_actions)
                
                if deterministic:
                    std.zero_()
                else:
                    std = std.clamp(min=eval_min_sampling_denoising_std)
                
                noise_sample.clamp_(-eval_randn_clip_value, eval_randn_clip_value)
                
                current_actions = mean + std * noise_sample
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        final_action_clip_value = getattr(self, 'final_action_clip_value', 1.0)
        if final_action_clip_value is not None:
            current_actions.clamp_(-final_action_clip_value, final_action_clip_value)

        final_actions = self.denorm_odo(current_actions)

        out: Dict[str, Any] = {"pred_traj": final_actions}
        if bool(getattr(self.config, "record_delta_pi_curve", False)) and hasattr(self, "_delta_pi_curve"):
            if len(self._delta_pi_curve) > 0:
                curve = torch.stack(self._delta_pi_curve, dim=0)  # [K]
                out["delta_pi_curve"] = curve
                out["delta_pi_mean"] = curve.mean()
        return BatchFeature(data=out)

    def sample_chain(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ):
        """
        Generates the full denoising chain and the final trajectory.
        This method reuses the logic from get_action but stores intermediate steps.

        Args:
            vl_features (torch.Tensor): Vision-language features from the backbone.
            his_traj_features (torch.Tensor): Encoded historical trajectory features.
            ego_status_features (torch.Tensor): Encoded ego status features.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start from.
                If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The full denoising chain as a tensor of shape (B, K+1, H, D).
                - The final, denormalized trajectory of shape (B, H, D).
        """
        B, D = vl_features.shape[0], self.config.action_dim
        device, dtype = vl_features.device, vl_features.dtype
        
        vl_features = self.feature_encoder(vl_features)
        
        his_traj_features = self.his_traj_encoder(
            his_traj_features.unsqueeze(1)               
        ).repeat(1, self.config.action_horizon, 1) 

        ego_status_features = self.ego_status_encoder(
            ego_status_features       
        )

        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )
        denoising_chain = [current_actions.clone()]

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.config.flow_cfg.num_timestep_buckets)
                t_batch = torch.full((B,), idx, device=device, dtype=torch.long)
                
                action_features = self.action_encoder(current_actions, t_batch)
                if hasattr(self, 'position_embedding'):
                    action_features += self.position_embedding(torch.arange(self.config.action_horizon, device=device))
                
                vl_features_mean = vl_features.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
                fused_input = self.fusion_projector(
                    torch.cat((his_traj_features, vl_features_mean, action_features), dim=2)
                )
                
                model_output = self.model(fused_input, vl_features, ego_status_features, t_batch)
                pred = self.action_decoder(model_output)
                
                pred_flow = pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred
                current_actions = current_actions + dt * pred_flow
                denoising_chain.append(current_actions.clone())

        elif self.config.sampling_method in ['ddpm', 'ddim']:
            if self.config.sampling_method == 'ddpm':
                step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
                timesteps = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            else: 
                timesteps = self.ddim_t
            
            for i, t_int in enumerate(timesteps):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device) if self.config.sampling_method == 'ddim' else t_batch

                mean, logvar, _ = self.p_mean_variance(
                    current_actions, t_batch, index_batch, vl_features, his_traj_features, ego_status_features, deterministic
                )

                std = torch.exp(0.5 * logvar).to(dtype)
                noise_sample = torch.randn_like(current_actions)

                if self.config.sampling_method == 'ddim':
                    if deterministic:
                        std.zero_()
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                else: # ddpm
                    if deterministic and t_int == 0:
                        std = torch.zeros_like(std)
                    elif deterministic:
                        std = std.clamp(min=1e-3)
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                
                if hasattr(self, 'randn_clip_value') and self.randn_clip_value is not None:
                    noise_sample = noise_sample.clamp_(-self.randn_clip_value, self.randn_clip_value)

                current_actions = mean + std * noise_sample
                
                if i == len(timesteps) - 1 and hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None:
                    current_actions = current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)
                
                denoising_chain.append(current_actions.clone())
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        final_actions = self.denorm_odo(current_actions)
        chain_tensor = torch.stack(denoising_chain, dim=1)
        
        return chain_tensor.detach(), final_actions.detach()

    def get_logprobs(
        self,
        vl_features: torch.Tensor,
        his_traj_features: torch.Tensor,
        ego_status_features: torch.Tensor,
        chains: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """Calculates the log probability of a full denoising chain."""
        B, K1, H, D = chains.shape
        num_denoising_steps = K1 - 1
        
        vl_features = self.feature_encoder(vl_features)

        his_traj_features = self.his_traj_encoder(
            his_traj_features.unsqueeze(1)          
        ).repeat(1, self.config.action_horizon, 1) 

        ego_status_features = self.ego_status_encoder(
            ego_status_features      
        )

        conditioning_embeds = {
            'vl_features': vl_features,
            'his_traj_features': his_traj_features,
            'ego_status_features': ego_status_features
        }
        
        batched_conditioning = {}
        for key, value in conditioning_embeds.items():
            batched_conditioning[key] = value.unsqueeze(1).repeat(
                1, num_denoising_steps, *(1,) * (value.ndim - 1)
            ).flatten(0, 1)

        if self.config.sampling_method == 'ddim':
            t_single = self.ddim_t[-num_denoising_steps:]
            indices_single = torch.arange(
                start=self.ddim_steps - num_denoising_steps,
                end=self.ddim_steps,
                device=chains.device
            )
            indices_batch = indices_single.repeat(B)
        else:
            t_single = torch.arange(start=K1 - 2, end=-1, step=-1, device=chains.device)
            indices_batch = None 
            
        t_batch = t_single.repeat(B)

        x_t = chains[:, :-1].reshape(-1, H, D)
        x_t_minus_1 = chains[:, 1:].reshape(-1, H, D)

        mean, logvar, _ = self.p_mean_variance(
            x_t, t_batch, indices_batch,
            batched_conditioning['vl_features'],
            batched_conditioning['his_traj_features'],
            batched_conditioning['ego_status_features'],
            deterministic=deterministic
        )

        std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(x_t_minus_1)
        
        return log_prob

    def forward_grpo(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: int = 8,
        deterministic=False,
        bc_coeff: float = 0.1,
        use_bc_loss: bool = True
    ) -> BatchFeature:
        """Computes the Diffusion-GRPO loss."""
        self.set_frozen_modules_to_eval_mode()
        B = vl_features.shape[0]
        G = sample_time 

        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)

        chains, trajs = self.sample_chain(
            vl_features_rep, his_traj_rep, status_feature_rep, deterministic=False
        )

        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        unique_tokens = set(tokens_list)
        metric_cache = {}
        for token in unique_tokens:
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, 'rb') as f:
                metric_cache[token] = pickle.load(f)
        rewards = self.reward_fn(trajs, tokens_rep, metric_cache)

        rewards_matrix = rewards.view(B, G)
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        std_r = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_matrix - mean_r) / std_r).view(-1).detach()
        
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        num_denoising_steps = chains.shape[1] - 1
        denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
        discount = (self.gamma_denoising ** (num_denoising_steps - denoising_indices - 1))
    
        
        adv_steps = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)  
        discount = discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)  
        adv_weighted_flat = (adv_steps * discount).reshape(-1)             

        log_probs = self.get_logprobs(vl_features_rep, his_traj_rep, status_feature_rep, chains, deterministic=False)
        log_probs = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        
        policy_loss = -torch.mean(log_probs * adv_weighted_flat)
        total_loss = policy_loss

        bc_loss = 0.0
        if use_bc_loss:
            with torch.no_grad():
                teacher_chains, _ = self.old_policy.sample_chain(
                    vl_features, action_input.his_traj, action_input.status_feature, deterministic=False
                )
            bc_logp = self.get_logprobs(vl_features, action_input.his_traj, action_input.status_feature, teacher_chains, deterministic=False)
            bc_logp = bc_logp.clamp(min=-5, max=2)
            K_steps = chains.shape[1] - 1
            bc_logp = bc_logp.view(-1, K_steps, chains.shape[2], chains.shape[3]).mean(dim=[1,2,3])
            bc_loss = -bc_logp.mean()
            total_loss = total_loss + bc_coeff * bc_loss

        return BatchFeature(data={"loss": total_loss, "reward": rewards.mean(), "policy_loss": policy_loss, "bc_loss": bc_loss})

    def norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalizes trajectory coordinates and heading to the range [-1, 1]."""
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)
    
    def denorm_odo(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    def reward_fn(
        self,
        pred_traj: torch.Tensor,
        tokens_list,
        cache_dict,
    ) -> torch.Tensor:
        """Calculates PDM scores for a batch of predicted trajectories."""
        pred_np = pred_traj.detach().cpu().numpy()
        rewards = []
        for i, token in enumerate(tokens_list):
            trajectory = Trajectory(pred_np[i])
            metric_cache = cache_dict[token]
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.simulator.proposal_sampling,
                simulator=self.simulator,
                scorer=self.train_scorer,
            )
            rewards.append(asdict(pdm_result)["score"])
        return torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

#https://github.com/irom-princeton/dppo/blob/cc7234ad7ff39a8f32de3af903606723a16f0648/model/diffusion/eta.py#L12
class EtaFixed(nn.Module):

    def __init__(
        self,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):
        super().__init__()
        self.eta_logit = nn.Parameter(torch.ones(1))
        self.min = min_eta
        self.max = max_eta

        self.eta_logit.data = torch.atanh(
            torch.tensor([2 * (base_eta - min_eta) / (max_eta - min_eta) - 1])
        )

    def __call__(self, x):
        """Match input batch size, but do not depend on input"""
        B = len(x)
        device = x.device
        eta_normalized = torch.tanh(self.eta_logit)

        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return torch.full((B, 1), eta.item()).to(device)
