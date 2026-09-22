import hashlib
from time import sleep

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Any, List
from navsim.common.dataclasses import Trajectory
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory

def _rowwise_isin(tensor_1: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    matches = (tensor_1[:, None] == target_tensor)
    
    return torch.sum(matches, dim=1, dtype=torch.bool)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(
        self,
        agent: AbstractAgent,
        for_viz=False,
        export_all_proposals=False,
        gradient_sensitivity_config=None,
    ):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file=None
        self.for_viz = for_viz
        self.export_all_proposals = export_all_proposals
        self.gradient_sensitivity_config = gradient_sensitivity_config or {}

    @staticmethod
    def _smooth_xy_gradient(xy_gradient: Tensor, sigma: float, kernel_size: int) -> Tensor:
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("smooth_kernel_size must be an odd integer >= 3")
        offsets = torch.arange(
            kernel_size, device=xy_gradient.device, dtype=xy_gradient.dtype
        ) - kernel_size // 2
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = (kernel / kernel.sum()).repeat(2, 1, 1)
        flattened = xy_gradient.reshape(-1, xy_gradient.shape[-2], 2).transpose(1, 2)
        padded = F.pad(flattened, (kernel_size // 2, kernel_size // 2), mode="reflect")
        return F.conv1d(padded, kernel, groups=2).transpose(1, 2).reshape_as(xy_gradient)

    @staticmethod
    def _poses_from_xy(base_proposals: Tensor, xy_delta: Tensor) -> Tensor:
        updated_xy = base_proposals[..., :2] + xy_delta
        origin = torch.zeros_like(updated_xy[..., :1, :])
        segments = torch.diff(torch.cat((origin, updated_xy), dim=-2), dim=-2)
        recomputed_heading = torch.atan2(
            segments[..., 1], segments[..., 0]
        ).unsqueeze(-1)
        has_displacement = xy_delta.abs().amax(dim=(-2, -1), keepdim=True) > 0
        heading = torch.where(
            has_displacement, recomputed_heading, base_proposals[..., 2:3]
        )
        return torch.cat((updated_xy, heading), dim=-1)

    @staticmethod
    def _normalize_max_xy(xy_direction: Tensor, epsilon: float) -> Tensor:
        max_norm = torch.linalg.vector_norm(xy_direction, dim=-1).amax(dim=-1)
        return xy_direction / max_norm.clamp_min(epsilon)[..., None, None]

    @staticmethod
    def _random_xy_direction(
        reference: Tensor, tokens: List[str], target: str, gradient_type: str
    ) -> Tensor:
        random_direction = torch.empty_like(reference)
        for batch_idx, token in enumerate(tokens):
            for proposal_idx in range(reference.shape[1]):
                seed_text = f"{token}:{proposal_idx}:{target}:{gradient_type}"
                seed = int.from_bytes(
                    hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "little"
                )
                generator = torch.Generator(device=reference.device).manual_seed(seed)
                random_direction[batch_idx, proposal_idx] = torch.randn(
                    reference.shape[-2:],
                    dtype=reference.dtype,
                    device=reference.device,
                    generator=generator,
                )
        return random_direction

    def _compute_gradient_sensitivity(self, predictions, tokens: List[str]):
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "Gradient sensitivity requires trainer.params.inference_mode=false"
            )

        metric_keys = {
            "NC": "no_at_fault_collisions",
            "DAC": "drivable_area_compliance",
            "TTC": "time_to_collision_within_bound",
        }
        proposals = predictions["proposals"].detach().requires_grad_(True)
        scene_features = predictions["scorer_scene_features"].detach()
        ego_token = predictions["scorer_ego_token"].detach()
        model = self.agent._drivor_model

        with torch.enable_grad():
            pred_logit = model.score_proposals(
                proposals, scene_features, ego_token, detach_proposals=False
            )[0]
            gradients = {}
            gradient_calls = len(metric_keys) * proposals.shape[1]
            call_index = 0
            for short_name, metric_name in metric_keys.items():
                metric_scores = torch.sigmoid(pred_logit[metric_name])
                metric_gradients = torch.empty_like(proposals)
                for proposal_idx in range(proposals.shape[1]):
                    call_index += 1
                    full_gradient = torch.autograd.grad(
                        metric_scores[:, proposal_idx].sum(),
                        proposals,
                        retain_graph=call_index < gradient_calls,
                    )[0]
                    metric_gradients[:, proposal_idx] = full_gradient[:, proposal_idx]
                gradients[short_name] = metric_gradients.detach()

        epsilon = float(self.gradient_sensitivity_config.get("epsilon", 1e-12))
        displacements = [
            float(value)
            for value in self.gradient_sensitivity_config.get(
                "displacements", [0.1, 0.3, 0.5, 0.7]
            )
        ]
        sigma = float(self.gradient_sensitivity_config.get("smooth_sigma", 1.0))
        kernel_size = int(self.gradient_sensitivity_config.get("smooth_kernel_size", 5))
        if sigma <= 0:
            raise ValueError("smooth_sigma must be positive")
        if any(displacement <= 0 for displacement in displacements):
            raise ValueError("gradient displacements must all be positive")
        base_proposals = proposals.detach()
        variants = {}
        gradient_stats = {}
        for short_name, gradient in gradients.items():
            flat_gradient = gradient.flatten(start_dim=2)
            gradient_norm = torch.linalg.vector_norm(flat_gradient, dim=-1)
            gradient_stats[short_name] = {
                "norm": gradient_norm.cpu(),
                "xy_norm": torch.linalg.vector_norm(
                    gradient[..., :2].flatten(start_dim=2), dim=-1
                ).cpu(),
                "heading_norm": torch.linalg.vector_norm(
                    gradient[..., 2], dim=-1
                ).cpu(),
            }
            raw_xy = gradient[..., :2]
            directions = {
                "raw": self._normalize_max_xy(raw_xy, epsilon),
                "smooth": self._normalize_max_xy(
                    self._smooth_xy_gradient(raw_xy, sigma, kernel_size), epsilon
                ),
            }
            for gradient_type, up_direction in directions.items():
                random_direction = self._random_xy_direction(
                    up_direction, tokens, short_name, gradient_type
                )
                if gradient_type == "smooth":
                    random_direction = self._smooth_xy_gradient(
                        random_direction, sigma, kernel_size
                    )
                random_direction = self._normalize_max_xy(random_direction, epsilon)
                for displacement in displacements:
                    displacement_key = f"{displacement:.12g}"
                    variants.setdefault(displacement_key, {}).setdefault(
                        short_name, {}
                    )[gradient_type] = {
                        "up": self._poses_from_xy(
                            base_proposals, displacement * up_direction
                        ).cpu(),
                        "down": self._poses_from_xy(
                            base_proposals, -displacement * up_direction
                        ).cpu(),
                        "random": self._poses_from_xy(
                            base_proposals, displacement * random_direction
                        ).cpu(),
                    }

        return variants, gradient_stats

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)

        if type(loss_dict) is dict:
            for key,value in loss_dict.items():
                self.log(f"{logging_prefix}/"+key, value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return loss_dict["loss"]
        else:
            return loss_dict

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
        if 'drivor' in self.agent.name() or "DrivoR" in self.agent.name():
            features, targets = batch
            # score,best_score=self.agent.inference(features, targets)
            predictions = self.agent.forward(features)
            all_chosen_trajectories = predictions["trajectory"][:,None]
            all_proposed_trajectories = predictions["proposals"]
            final_score, fake_best_score, proposal_scores, l2, trajectoy_scores = self.agent.compute_score(targets, all_chosen_trajectories)
            _, best_score, all_proposal_scores, _, _ = self.agent.compute_score(targets, all_proposed_trajectories)
            mean_score=proposal_scores.mean()

            logging_prefix="val"
            if "pdm_score" in predictions:
                pdm_score = predictions["pdm_score"]
                best_pred_score_values = pdm_score[torch.arange(len(pdm_score)), torch.argmax(pdm_score, dim=1)]
                score_error = torch.abs(best_pred_score_values - proposal_scores).mean()
                self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                
                best_pred_score_index = torch.argmax(pdm_score, dim=1)
                best_real_score_index = torch.argmax(all_proposal_scores, dim=1)
                score_hit_rate = torch.mean(best_pred_score_index == best_real_score_index, dtype=torch.float32)

                best_possible_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_real_score_index]
                best_actual_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_pred_score_index]
                lost_score = torch.mean(best_possible_scores - best_actual_scores)
                self.log(f"{logging_prefix}/score_hit_rate", score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/lost_score", lost_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

                top_5_indices_real = torch.topk(all_proposal_scores, k=5, dim=1).indices
                top_5_score_hit_rate = _rowwise_isin(best_pred_score_index, top_5_indices_real).mean(dtype=torch.float32)
                self.log(f"{logging_prefix}/top_5_score_hit_rate", top_5_score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
            self.log(f"{logging_prefix}/score", final_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            collision=trajectoy_scores[:,0].mean()
            self.log(f"{logging_prefix}/collision", collision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            drivable_area_compliance=trajectoy_scores[:,1].mean()
            self.log(f"{logging_prefix}/dac", drivable_area_compliance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            ego_progress=trajectoy_scores[:,2].mean()
            self.log(f"{logging_prefix}/progress", ego_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            time_to_collision_within_bound=trajectoy_scores[:,3].mean()
            self.log(f"{logging_prefix}/ttc", time_to_collision_within_bound, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            comfort=trajectoy_scores[:,4].mean()
            self.log(f"{logging_prefix}/comfort", comfort, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return final_score
        else:
            return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
    
    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Used during the multi-gpu proccessing to parallelize the prediction of trajectories.
        NOTE: requires append_token_to_batch=True in the dataset used to instantiate the trainer.
        """
        return self.predict_step_drivor(batch, batch_idx)

    def predict_step_drivor(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]
            if self.export_all_proposals:
                proposals = predictions["proposals"]
                predicted_pdm_scores = predictions["pdm_score"]
                predicted_subscores = {
                    metric_name: torch.sigmoid(metric_logits)
                    for metric_name, metric_logits in predictions["pred_logit"].items()
                }
                if predicted_pdm_scores.ndim != 2:
                    raise ValueError(
                        "Expected pdm_score shape [B, N], "
                        f"got {tuple(predicted_pdm_scores.shape)}"
                    )
                expected_shape = tuple(predicted_pdm_scores.shape)
                if tuple(proposals.shape[:2]) != expected_shape:
                    raise ValueError(
                        f"Expected proposals prefix {expected_shape}, "
                        f"got {tuple(proposals.shape[:2])}"
                    )
                for metric_name, metric_scores in predicted_subscores.items():
                    if tuple(metric_scores.shape) != expected_shape:
                        raise ValueError(
                            f"Expected pred_logit[{metric_name!r}] shape {expected_shape}, "
                            f"got {tuple(metric_scores.shape)}"
                        )
                selected_indices = torch.argmax(predicted_pdm_scores, dim=1)
            if self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                ego_status = features["ego_status"]
        gradient_variants = gradient_stats = None
        if self.gradient_sensitivity_config.get("enabled", False):
            if not self.export_all_proposals:
                raise ValueError("Gradient sensitivity requires export_all_proposals=true")
            gradient_variants, gradient_stats = self._compute_gradient_sensitivity(
                predictions, tokens
            )
        result = {}
        for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            if self.for_viz:
                proposal_list = [proposal_list[index].cpu().numpy() for proposal_list in all_proposed_trajectories]
                result[token] = {
                    'trajectory': proposal, 
                    'all_proposals': proposal_list, 
                    'all_proposal_scores': final_scores[index],
                    'high_level_command': ego_status[index]
                }
            else:
                result[token] = {'trajectory': proposal}
                if self.export_all_proposals:
                    result[token]['proposals'] = [
                        Trajectory(poses) for poses in proposals[index].cpu().numpy()
                    ]
                    selected_idx = int(selected_indices[index].item())
                    proposal_count = proposals.shape[1]
                    if not 0 <= selected_idx < proposal_count:
                        raise ValueError(
                            f"Selected proposal index {selected_idx} is outside [0, {proposal_count})"
                        )
                    result[token]['selected_proposal_idx'] = selected_idx
                    result[token]['predicted_subscores'] = {
                        metric_name: metric_scores[index].detach().cpu().numpy()
                        for metric_name, metric_scores in predicted_subscores.items()
                    }
                    result[token]['predicted_pdm_scores'] = (
                        predicted_pdm_scores[index].detach().cpu().numpy()
                    )
                    if gradient_variants is not None:
                        result[token]["gradient_variants"] = {
                            displacement: {
                                metric: {
                                    gradient_type: {
                                        direction: [
                                            Trajectory(variant_poses)
                                            for variant_poses in direction_poses[index].numpy()
                                        ]
                                        for direction, direction_poses in type_variants.items()
                                    }
                                    for gradient_type, type_variants in metric_variants.items()
                                }
                                for metric, metric_variants in eta_variants.items()
                            }
                            for displacement, eta_variants in gradient_variants.items()
                        }
                        result[token]["gradient_stats"] = {
                            metric: {
                                stat_name: stat_values[index].numpy()
                                for stat_name, stat_values in metric_stats.items()
                            }
                            for metric, metric_stats in gradient_stats.items()
                        }
        return result
