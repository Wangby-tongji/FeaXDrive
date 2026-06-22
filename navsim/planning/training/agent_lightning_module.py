import pytorch_lightning as pl
import torch

from torch import Tensor
from typing import Dict, Tuple,Any

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        #prediction = self.agent.forward(features,targets)
        loss = self.agent.compute_loss(features, targets, prediction)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()


class AgentLightningDiT(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        if logging_prefix == 'train':
            # Source-compatible behavior:
            #   - Non-GRPO IL/dyn training:
            #       agent.compute_loss(...) returns a scalar Tensor, because
            #       FeaXDriveAgent.compute_loss returns predictions.loss.
            #   - FA-GRPO training:
            #       agent.compute_loss(...) returns a BatchFeature-like object
            #       with .loss/.reward/.policy_loss/.bc_loss fields.
            #
            # The old unified AgentLightningDiT assumed the GRPO object path
            # for all modes, causing AttributeError: 'Tensor' object has no
            # attribute 'loss' during MODE=dyn.
            loss_output = self.agent.compute_loss(features, targets, prediction)

            if torch.is_tensor(loss_output):
                loss = loss_output
                self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

                # Log auxiliary IL/dyn quantities from the raw planner output
                # when available. These keys come from the diffusion planner's
                # BatchFeature output during x0/dyn training.
                for src_key, log_key in [
                    ("loss_base", "loss_base"),
                    ("loss_dyn", "dyn_loss"),
                    ("loss_proj", "proj_loss"),
                    ("loss_pi", "pi_loss"),
                    ("dyn_kappa_violation_rate", "dyn_kappa_violation_rate"),
                    ("dyn_kappa_adapt_violation_rate", "dyn_kappa_adapt_violation_rate"),
                    ("dyn_a_lat_violation_rate", "dyn_a_lat_violation_rate"),
                ]:
                    try:
                        val = prediction[src_key] if src_key in prediction else getattr(prediction, src_key, None)
                    except Exception:
                        val = getattr(prediction, src_key, None)
                    if val is not None and torch.is_tensor(val):
                        self.log(f"{logging_prefix}/{log_key}", val, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                predictions = loss_output
                loss = predictions.loss
                reward = predictions.reward
                policy_loss = predictions.policy_loss
                bc_loss = predictions.bc_loss
                dyn_loss = getattr(predictions, "dyn_loss", None)
                dyn_kappa_violation_rate = getattr(predictions, "dyn_kappa_violation_rate", None)
                dyn_kappa_adapt_violation_rate = getattr(predictions, "dyn_kappa_adapt_violation_rate", None)
                dyn_a_lat_violation_rate = getattr(predictions, "dyn_a_lat_violation_rate", None)
                self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/reward", reward, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                if dyn_loss is not None:
                    self.log(f"{logging_prefix}/dyn_loss", dyn_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                if dyn_kappa_violation_rate is not None:
                    self.log(f"{logging_prefix}/dyn_kappa_violation_rate", dyn_kappa_violation_rate, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                if dyn_kappa_adapt_violation_rate is not None:
                    self.log(f"{logging_prefix}/dyn_kappa_adapt_violation_rate", dyn_kappa_adapt_violation_rate, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                if dyn_a_lat_violation_rate is not None:
                    self.log(f"{logging_prefix}/dyn_a_lat_violation_rate", dyn_a_lat_violation_rate, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            prediction = self.agent.forward(features,targets)
            loss = self.agent.compute_loss(features, targets, prediction)
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        #print(batch_idx)
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()