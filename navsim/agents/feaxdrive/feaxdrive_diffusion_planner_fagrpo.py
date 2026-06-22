"""FA-GRPO-only diffusion planner for FeaXDrive.

This module isolates FA-GRPO training reward logic from the tested IL/dyn and
final evaluation chains. It reuses the standard FeaXDrive diffusion planner
for model architecture, denoising, drivable guidance, and checkpoint format,
but overrides GRPO initialization and reward scoring to use the FA-GRPO scorer.

Active only when the agent selects planner_variant="fagrpo".
"""

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score_feaxdrive_fagrpo import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.feaxdrive_fagrpo_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .feaxdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


@dataclass
class GRPOConfigFAGRPO:
    """Configuration specific to FA-GRPO training."""

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
    bc_coeff: float = 0.1

    metric_cache_path: str = "/path/to/metric_cache_train"
    reference_policy_checkpoint: str = "/path/to/IL_Model.ckpt"
    scorer_config: PDMScorerConfig = field(
        default_factory=lambda: PDMScorerConfig(
            progress_weight=10.0,
            ttc_weight=5.0,
            comfortable_weight=10.0,
        )
    )


@dataclass
class FeaXDriveFAGRPODiffusionPlannerConfig(ReCogDriveDiffusionPlannerConfig):
    """Standard FeaXDrive planner config with FA-GRPO reward defaults."""

    grpo_cfg: GRPOConfigFAGRPO = field(default_factory=GRPOConfigFAGRPO)


class FeaXDriveFAGRPODiffusionPlanner(ReCogDriveDiffusionPlanner):
    """FeaXDrive diffusion planner with isolated FA-GRPO reward path."""

    config_class = FeaXDriveFAGRPODiffusionPlannerConfig

    def _init_grpo(self, cfg: GRPOConfigFAGRPO):
        """Initialize FA-GRPO components and reference policy."""
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
        self.bc_coeff = cfg.bc_coeff

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
                    print(
                        f"Skip loading '{k}' -> '{k2}' "
                        f"(checkpoint shape {tuple(v.shape)} vs model shape {tuple(model_dict.get(k2, v).shape)})"
                    )
            self.load_state_dict(filtered_ckpt, strict=True)
        except FileNotFoundError:
            print(f"Warning: FA-GRPO reference checkpoint not found at {cfg.reference_policy_checkpoint}. Skipping loading.")

        self.old_policy = copy.deepcopy(self)
        self.old_policy.eval()
        for param in self.old_policy.parameters():
            param.requires_grad = False

    def forward_grpo(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: int = 8,
        deterministic=False,
        bc_coeff: Optional[float] = None,
        use_bc_loss: bool = True,
    ) -> BatchFeature:
        if bc_coeff is None:
            bc_coeff = float(getattr(self, "bc_coeff", 0.1))
        return super().forward_grpo(
            vl_features=vl_features,
            action_input=action_input,
            tokens_list=tokens_list,
            sample_time=sample_time,
            deterministic=deterministic,
            bc_coeff=bc_coeff,
            use_bc_loss=use_bc_loss,
        )

    def reward_fn(
        self,
        pred_traj: torch.Tensor,
        tokens_list,
        cache_dict,
    ) -> torch.Tensor:
        """Calculate FA-GRPO PDM reward for a batch of predicted trajectories."""
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


FeaXDriveDiffusionPlannerFAGRPO = FeaXDriveFAGRPODiffusionPlanner
FeaXDriveDiffusionPlannerConfigFAGRPO = FeaXDriveFAGRPODiffusionPlannerConfig
