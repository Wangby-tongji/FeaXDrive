from typing import Any, List, Dict, Optional, Union
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .recogdrive_features import ReCogDriveFeatureBuilder ,TrajectoryTargetBuilder
from .recogdrive_backbone import RecogDriveBackbone
from .recogdrive_diffusion_planner_entry_v4drive import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single', 
        vlm_type: Optional[str] = 'internvl', 
        dit_type: Optional[str] = 'small', 
        sampling_method: Optional[str] = 'ddim', 
        diffusion_pred_type: Optional[str] = 'eps', 
        cache_mode: bool = False, 
        cache_hidden_state: bool = True, 
        lr: float = 1e-4,
        grpo: bool = False,
        metric_cache_path: Optional[str] = '', 
        reference_policy_checkpoint: Optional[str] = '', 
        vlm_size: Optional[str] = 'small', 
        train_backbone: bool = False,

        # ---- Dynamics-feasible projection & constraint-consistency training ----
        use_constraint_projection: bool = False,
        proj_dt: float = 0.5,
        proj_kappa_max: Optional[float] = None,
        proj_kappa_geo_max: Optional[float] = None,
        proj_use_kappa_adapt: bool = False,
        proj_a_lat_max: Optional[float] = None,
        proj_smooth_ksize: int = 5,
        proj_step_size: float = 0.1,
        proj_lambda_kappa: float = 1.0,
        proj_lambda_a_lat: float = 1.0,
        proj_iters: int = 1,
        proj_keep_start: bool = True,
        lambda_dyn: float = 0.0,
        lambda_proj: float = 0.0,
        lambda_proj_supervise: float = 0.0,
        proj_stop_grad: bool = True,
        record_delta_pi_curve: bool = False,

        # ---- Drivable-area x0-guidance (inference-time) ----
        use_drivable_guidance: bool = False,
        drivable_guidance_last_k: int = 3,
        drivable_guidance_step_size: float = 0.05,
        drivable_guidance_margin_m: float = 0.30,
        drivable_guidance_beta_m: float = 0.20,
        drivable_guidance_oob_sdf_m: float = -10.0,
        drivable_guidance_power: float = 1.0,
        drivable_guidance_heading_scale: float = 0.05,
        drivable_guidance_only_offroad: bool = False,
        drivable_guidance_offroad_eps_m: float = 0.0,

        # ---- Online drivable SDF rasterization (ego-frame ROI) ----
        drivable_sdf_x_min: float = -10.0,
        drivable_sdf_x_max: float = 80.0,
        drivable_sdf_y_min: float = -30.0,
        drivable_sdf_y_max: float = 30.0,
        drivable_sdf_resolution: float = 0.20,
        drivable_sdf_clip: float = 50.0,
        drivable_sdf_source: str = 'metric_cache',
        drivable_sdf_map_radius: float = 100.0,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self._lr = lr
        self.grpo = grpo
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device
        if not self.cache_hidden_state and not self.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_path or not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            self.backbone = RecogDriveBackbone(
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=device
            )

            if not self.train_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True

        if self.dit_type == "large":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=1536,sampling_method=sampling_method, pred_type=diffusion_pred_type)
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=384,sampling_method=sampling_method, pred_type=diffusion_pred_type)

        cfg.vlm_size = self.vlm_size

        # Plug projection/constraint settings into the diffusion planner config.
        cfg.use_constraint_projection = use_constraint_projection
        cfg.proj_dt = proj_dt
        cfg.proj_kappa_max = proj_kappa_max
        cfg.proj_kappa_geo_max = proj_kappa_geo_max
        cfg.proj_use_kappa_adapt = proj_use_kappa_adapt
        cfg.proj_a_lat_max = proj_a_lat_max
        cfg.proj_smooth_ksize = proj_smooth_ksize
        cfg.proj_step_size = proj_step_size
        cfg.proj_lambda_kappa = proj_lambda_kappa
        cfg.proj_lambda_a_lat = proj_lambda_a_lat
        cfg.proj_iters = proj_iters
        cfg.proj_keep_start = proj_keep_start
        cfg.lambda_dyn = lambda_dyn
        cfg.lambda_proj = lambda_proj
        cfg.lambda_proj_supervise = lambda_proj_supervise
        cfg.proj_stop_grad = proj_stop_grad
        cfg.record_delta_pi_curve = record_delta_pi_curve
        cfg.record_delta_pi_curve = record_delta_pi_curve

        # Drivable guidance hyperparameters (consumed inside diffusion planner).
        cfg.use_drivable_guidance = use_drivable_guidance
        cfg.drivable_guidance_last_k = drivable_guidance_last_k
        cfg.drivable_guidance_step_size = drivable_guidance_step_size
        cfg.drivable_guidance_margin_m = drivable_guidance_margin_m
        cfg.drivable_guidance_beta_m = drivable_guidance_beta_m
        cfg.drivable_guidance_oob_sdf_m = drivable_guidance_oob_sdf_m
        cfg.drivable_guidance_power = drivable_guidance_power
        cfg.drivable_guidance_heading_scale = drivable_guidance_heading_scale
        cfg.drivable_guidance_only_offroad = drivable_guidance_only_offroad
        cfg.drivable_guidance_offroad_eps_m = drivable_guidance_offroad_eps_m
        # Keep SDF rasterization parameters on the agent so evaluation scripts can
        # build per-token SDF contexts without hard-coding.
        self.drivable_sdf_cfg = {
            "x_min": drivable_sdf_x_min,
            "x_max": drivable_sdf_x_max,
            "y_min": drivable_sdf_y_min,
            "y_max": drivable_sdf_y_max,
            "resolution": drivable_sdf_resolution,
            "sdf_clip": drivable_sdf_clip,
        }
        # Source for inference-time SDF construction:
        #   metric_cache: use the evaluator's serialized PDMDrivableMap (paper reproduction default)
        #   scene_map/data_map: rebuild from NAVSIM scene map_api and current ego state
        self.drivable_sdf_source = drivable_sdf_source
        self.drivable_sdf_map_radius = drivable_sdf_map_radius

        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda()
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def set_drivable_sdf_context(self, ctx: Optional[Dict[str, object]]) -> None:
        """Attach per-scenario drivable SDF context for inference-time guidance."""
        try:
            if hasattr(self, "action_head") and hasattr(self.action_head, "set_drivable_sdf_context"):
                self.action_head.set_drivable_sdf_context(ctx)
        except Exception:
            return

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self.checkpoint_path:
            # flag
            ckpt_obj = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            ckpt = ckpt_obj["state_dict"]
            # ckpt = torch.load(self.checkpoint_path, map_location="cpu")["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in ckpt.items():
                k2 = k[len("agent."):] if k.startswith("agent.") else k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            self.load_state_dict(filtered_ckpt, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=self.device,
            cache_mode=self.cache_mode,
        )]

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype

        history_trajectory = features["history_trajectory"].cuda()
        high_command_one_hot = features["high_command_one_hot"].cuda()
        
        
        if history_trajectory.ndim == 2: history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1: high_command_one_hot = high_command_one_hot.unsqueeze(0)

        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"].cuda()
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1: image_path_tensor = image_path_tensor.unsqueeze(0)
            image_paths = self._decode_paths_from_tensor(image_path_tensor)
            
            pixel_values_list = [load_image(path) for path in image_paths]
            
            num_patches_list = [p.shape[0] for p in pixel_values_list]
            pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()
            

            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_indices = torch.argmax(high_command_one_hot, dim=-1)
            command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

            questions = []
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                history_trajectory_sample = history_trajectory[i]
                command_str_sample = command_str_list[i]

                history_str = ' '.join([
                    f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                    f'{format_number(history_trajectory_sample[j, 1].item())}, '
                    f'{format_number(history_trajectory_sample[j, 2].item())})'
                    for j in range(history_trajectory_sample.shape[0])
                ])
                
                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_sample.upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")

            outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1: status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2: last_hidden_state = last_hidden_state.unsqueeze(0)

        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)

        if self.training and not self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            return self.action_head(last_hidden_state, action_inputs)
        elif self.training and self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
        else: 
            action_inputs = BatchFeature({"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype)})
            return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        # Optional: expose ΔΠ curve from the planner for logging during evaluation.
        self.last_delta_pi_curve = None
        self.last_delta_pi_mean = None
        try:
            if isinstance(predictions, BatchFeature):
                if "delta_pi_curve" in predictions:
                    c = predictions["delta_pi_curve"]
                    if torch.is_tensor(c):
                        self.last_delta_pi_curve = c.detach().cpu().tolist()
                    else:
                        self.last_delta_pi_curve = list(c)
                if "delta_pi_mean" in predictions:
                    m = predictions["delta_pi_mean"]
                    self.last_delta_pi_mean = float(m.detach().cpu().item()) if torch.is_tensor(m) else float(m)
        except Exception:
            pass

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))
        optimizer = build_from_configs(optim, optimizer_cfg, params=self.action_head.parameters())
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    pred_type: str = 'eps',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        pred_type=pred_type,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )
    
    return config
