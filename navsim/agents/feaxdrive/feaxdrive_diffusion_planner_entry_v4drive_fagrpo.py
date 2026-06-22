"""Compatibility wrapper for legacy FA-GRPO imports.

Canonical implementation:
    navsim.agents.feaxdrive.feaxdrive_diffusion_planner

FA-GRPO should be selected by agent/slurm config, not by maintaining a second
planner implementation.
"""

from .feaxdrive_diffusion_planner import *
