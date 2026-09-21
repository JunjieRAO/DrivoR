from time import sleep

import numpy as np
import pytorch_lightning as pl
import torch
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

    def _compute_gradient_sensitivity(self, predictions):
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

        normalize = bool(self.gradient_sensitivity_config.get("normalize", False))
        epsilon = float(self.gradient_sensitivity_config.get("epsilon", 1e-12))
        etas = [float(value) for value in self.gradient_sensitivity_config.get("etas", [1e-3])]
        base_proposals = proposals.detach()
        variants = {}
        gradient_stats = {}
        for short_name, gradient in gradients.items():
            flat_gradient = gradient.flatten(start_dim=2)
            gradient_norm = torch.linalg.vector_norm(flat_gradient, dim=-1)
            update_direction = gradient
            if normalize:
                update_direction = gradient / gradient_norm.clamp_min(epsilon)[..., None, None]

            gradient_stats[short_name] = {
                "norm": gradient_norm.cpu(),
                "xy_norm": torch.linalg.vector_norm(
                    gradient[..., :2].flatten(start_dim=2), dim=-1
                ).cpu(),
                "heading_norm": torch.linalg.vector_norm(
                    gradient[..., 2], dim=-1
                ).cpu(),
            }
            for eta in etas:
                eta_key = f"{eta:.12g}"
                variants.setdefault(eta_key, {})[short_name] = {
                    "up": (base_proposals + eta * update_direction).cpu(),
                    "down": (base_proposals - eta * update_direction).cpu(),
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
            gradient_variants, gradient_stats = self._compute_gradient_sensitivity(predictions)
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
                        result[token]["gradient_normalized"] = bool(
                            self.gradient_sensitivity_config.get("normalize", False)
                        )
                        result[token]["gradient_variants"] = {
                            eta: {
                                metric: {
                                    direction: [
                                        Trajectory(variant_poses)
                                        for variant_poses in direction_poses[index].numpy()
                                    ]
                                    for direction, direction_poses in metric_variants.items()
                                }
                                for metric, metric_variants in eta_variants.items()
                            }
                            for eta, eta_variants in gradient_variants.items()
                        }
                        result[token]["gradient_stats"] = {
                            metric: {
                                stat_name: stat_values[index].numpy()
                                for stat_name, stat_values in metric_stats.items()
                            }
                            for metric, metric_stats in gradient_stats.items()
                        }
        return result
