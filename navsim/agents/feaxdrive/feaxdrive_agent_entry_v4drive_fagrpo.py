"""FA-GRPO FeaXDrive agent entry.

This file is intentionally a thin wrapper around the official FeaXDrive
v4drive agent. It keeps FA-GRPO as a separate Hydra target while avoiding
drift between the standard IL agent and the FA-GRPO agent.

Important:
- This module only affects `agent=feaxdrive_v4drive_fagrpo_agent`.
- It does not affect eps/dt/dyn/drivedyn evaluation, which uses
  `feaxdrive_v4drive_agent`.
- It does not restore or depend on the deprecated `use_constraint_projection`.
- Dyn/FA training feasibility hyperparameters such as `proj_dt`,
  `proj_kappa_geo_max`, `proj_use_kappa_adapt`, and `lambda_dyn` are accepted
  by the base FeaXDrive agent and passed into the unified planner config.
"""

from typing import Any

from .feaxdrive_agent_entry_v4drive import FeaXDriveAgent as _BaseFeaXDriveAgent


class FeaXDriveAgent(_BaseFeaXDriveAgent):
    """FA-GRPO wrapper.

    Defaults `planner_variant` to ``"fagrpo"`` while preserving all arguments
    supported by the standard FeaXDrive agent. Training scripts may still pass
    ``planner_variant=fagrpo`` explicitly; this wrapper simply makes that the
    safe default for this Hydra target.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("planner_variant", "fagrpo")
        super().__init__(*args, **kwargs)


# Optional aliases for scripts that import a more explicit class name.
FeaXDriveFAGRPOAgent = FeaXDriveAgent
FeaXDriveAgentFAGRPO = FeaXDriveAgent
