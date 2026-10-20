from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from .drivor_model import DrivoRModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from .drivor_features import DrivoRTargetBuilder
from .drivor_features import DrivoRFeatureBuilder
import sys
from omegaconf import OmegaConf
import math
import warnings

class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - train {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - val {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(self, pl_module)
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = dict()
        val_metrics = dict()
        other_metrics = dict()
        for k,v in metrics.items():
            if "train/" in k:
                train_metrics[k]=v
            elif "val/" in k:
                val_metrics[k]=v
            else:
                other_metrics[k]=v
        print(f"\n###########  Epoch {trainer.current_epoch} ##########")
        for k,v in train_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in val_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in other_metrics.items():
            print(f"{k},{v:.3f}")
        print(f"###########\n")

class DrivoRAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: str = None,
            loss: nn.Module = None,
            progress_bar: bool = True,
            scheduler_args: dict = None,
            batch_size: int = 64,
            num_gpus: int = 1,
                train_metric_cache_path: str = None,
                perception_lr_scale: float = 1.0,
                use_ray: bool = True,
                checkpoint_perception_only: bool = False,
    ):
        super().__init__()
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self._checkpoint_perception_only = checkpoint_perception_only
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.loss = loss
        self.train_metric_cache_path = train_metric_cache_path
        self.perception_lr_scale = perception_lr_scale
        self.ray = use_ray
        self._score_backend_initialized = False


        cache_data=False

        if not cache_data:
            self._drivor_model = DrivoRModel(config)

    def _initialize_score_backend(self):
        if self._score_backend_initialized:
            return
        if self.ray:
            from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
            from nuplan.planning.utils.multithreading.worker_utils import worker_map
            self.worker = RayDistributedNoTorch(threads_per_node=8)
            self.worker_map = worker_map
        from .score_module.compute_navsim_score import get_scores
        cache_path = self.train_metric_cache_path or str(Path(os.environ["NAVSIM_EXP_ROOT"]) / "train_metric_cache")
        metric_cache = MetricCacheLoader(Path(cache_path))
        self.train_metric_cache_paths = metric_cache.metric_cache_paths
        self.test_metric_cache_paths = self.train_metric_cache_paths
        synthetic_paths = {}
        for index in range(5):
            path = Path(os.environ["NAVSIM_EXP_ROOT"]) / f"train_metric_synthetic_reaction_pdm_v1.0-{index}"
            if path.exists():
                synthetic_paths.update(MetricCacheLoader(path).metric_cache_paths)
        self.train_metric_cache_paths_synthetic = synthetic_paths
        self.test_metric_cache_paths_synthetic = synthetic_paths
        self.get_scores = get_scores
        self._score_backend_initialized = True


    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path:
            checkpoint = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {key.replace("agent._drivor_model", "_drivor_model"): value
                          for key, value in state_dict.items()}
            if self._checkpoint_perception_only:
                current_state = self.state_dict()
                perception_keys = {key for key in current_state
                                   if key.startswith(("_drivor_model.image_backbone.", "_drivor_model.lidar_backbone."))
                                   or key in ("_drivor_model.scene_embeds", "_drivor_model.lidar_scene_embeds")}
                missing = perception_keys - state_dict.keys()
                if not perception_keys or missing:
                    raise ValueError(f"Incomplete perception checkpoint; missing keys: {sorted(missing)}")
                current_state.update({key: state_dict[key] for key in perception_keys})
                state_dict = current_state
            else:
                current_state = self.state_dict()
                lane_keeping_keys = {
                    key for key in current_state
                    if key.startswith("_drivor_model.scorer.pred_score.lane_keeping.")
                }
                if lane_keeping_keys and current_state.keys() - state_dict.keys() == lane_keeping_keys:
                    if self._config.get("lk", 0.0) != 0.0:
                        raise ValueError(
                            "Checkpoint has no trained lane-keeping head; set agent.config.lk=0 "
                            "to load it for fine-tuning or legacy ranking."
                        )
                    state_dict.update({key: current_state[key] for key in lane_keeping_keys})
                    warnings.warn(
                        "Checkpoint has no lane-keeping head; initialized it from scratch with LK ranking disabled.",
                        UserWarning, stacklevel=2,
                    )
            self.load_state_dict(state_dict)

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        # return SensorConfig(
        #     cam_f0=[3],
        #     cam_l0=[3],
        #     cam_l1=[],
        #     cam_l2=[],
        #     cam_r0=[3],
        #     cam_r1=[],
        #     cam_r2=[],
        #     cam_b0=[3],
        #     lidar_pc=[],
        # )
        return SensorConfig(
            cam_f0=OmegaConf.to_object(self._config["cam_f0"]),
            cam_l0=OmegaConf.to_object(self._config["cam_l0"]),
            cam_l1=OmegaConf.to_object(self._config["cam_l1"]),
            cam_l2=OmegaConf.to_object(self._config["cam_l2"]),
            cam_r0=OmegaConf.to_object(self._config["cam_r0"]),
            cam_r1=OmegaConf.to_object(self._config["cam_r1"]),
            cam_r2=OmegaConf.to_object(self._config["cam_r2"]),
            cam_b0=OmegaConf.to_object(self._config["cam_b0"]),
            lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
        )
    
    def get_target_builders(self) :
        builders = [DrivoRTargetBuilder(config=self._config)]
        if self._drivor_model.geometry_mode != "off":
            from .scene_geometry.target_builder import SceneGeometryTargetBuilder
            builders.append(SceneGeometryTargetBuilder(self._config))
        return builders

    def get_feature_builders(self) :
        return [DrivoRFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._drivor_model(features)

    def compute_score(self, targets, proposals, test=True):
        self._initialize_score_backend()
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
            metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
        else:
            metric_cache_paths = self.test_metric_cache_paths
            metric_cache_paths_synthetic = self.test_metric_cache_paths_synthetic

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()
        safety_config = None
        if not test and self.loss is not None and (self.loss.dac_weight > 0 or self.loss.collision_weight > 0):
            safety_config = {
                "bounds": self.loss.sdf_bounds,
                "resolution": self.loss.sdf_resolution,
                "time_horizon": self._config.trajectory_sampling.time_horizon,
            }

        
        data_points = [
            {
                "token": metric_cache_paths[token] if token in metric_cache_paths else metric_cache_paths_synthetic[token],
                "poses": poses,
                "test": test,
                "safety_config": safety_config,
            }
            for token, poses in zip(targets["token"], proposals.cpu().numpy())
        ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            result = (final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas)
            if safety_config is not None:
                safety_targets = [res[4] for res in all_res]
                max_actors = max(item["obstacle_valid"].shape[1] for item in safety_targets)
                padded_corners, padded_valid = [], []
                for item in safety_targets:
                    count = item["obstacle_valid"].shape[1]
                    padded_corners.append(np.pad(item["obstacle_corners"], ((0, 0), (0, max_actors - count), (0, 0), (0, 0))))
                    padded_valid.append(np.pad(item["obstacle_valid"], ((0, 0), (0, max_actors - count))))
                batch_targets = {key: torch.as_tensor(np.stack([item[key] for item in safety_targets]), device=proposals.device)
                                 for key in ("drivable_sdf", "sdf_bounds", "vehicle")}
                batch_targets["obstacle_corners"] = torch.as_tensor(np.stack(padded_corners), device=proposals.device)
                batch_targets["obstacle_valid"] = torch.as_tensor(np.stack(padded_valid), device=proposals.device)
                result += (batch_targets,)
            return result

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        losses = self.loss(targets, pred, self._config, self.compute_score)
        if "scene_geometry" in pred:
            geometry = self.compute_geometry_metrics(targets, pred)
            weight = self._config.scene_geometry.loss_weight * getattr(self, "geometry_ramp", 0.0)
            losses.update(geometry)
            losses["geometry/weight"] = losses["loss"].new_tensor(weight)
            losses["loss"] = losses["loss"] + weight * geometry["geometry/loss"]
        return losses

    def set_geometry_progress(self, global_step, total_steps):
        cfg = self._config.get("scene_geometry", {})
        warmup = max(1, int(total_steps * cfg.get("warmup_fraction", 0.1)))
        self.geometry_ramp = min(1.0, global_step / warmup)

    def compute_geometry_metrics(self, targets, predictions):
        if "scene_geometry" not in predictions:
            return {}
        from .scene_geometry.geometry_loss import geometry_loss
        result = geometry_loss(predictions["scene_geometry"], targets)
        for kind in ("agent", "map"):
            result[f"geometry/{kind}_overflow"] = targets[f"sg_{kind}_overflow"].float().mean()
            result[f"geometry/{kind}_available"] = targets[f"sg_{kind}_available"].float().mean()
        return result

    def get_optimizers(self):

        global_batchsize = self.batch_size * self.num_gpus
        lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
        perception, planning = [], []
        for name, parameter in self._drivor_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(("image_backbone.", "lidar_backbone.")) or name in ("scene_embeds", "lidar_scene_embeds"):
                perception.append(parameter)
            else:
                planning.append(parameter)
        parameters = [{"params": planning, "lr": lr},
                      {"params": perception, "lr": lr * self.perception_lr_scale}]
        if self._lr_args["name"] == "Adam":
            optimizer = torch.optim.Adam(parameters, lr=lr)
        elif self._lr_args["name"] == "AdamW":
            optimizer = torch.optim.AdamW(parameters, lr=lr)
        else:
            raise NotImplementedError

        if self.scheduler_args is not None:

            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) *  self.scheduler_args.num_epochs)

            # classic cosine
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=T_max, 
            #     eta_min=0.0, last_epoch=-1
            # )

            # Ramp + cosine
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
            T_max_cosine = T_max - T_max_ramp
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max_cosine, 
                eta_min=0.0, last_epoch=-1
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_ramp, scheduler_cosine],
                milestones=[T_max_ramp],
            )           

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        else:
            return [optimizer]

    def get_training_callbacks(self):

        checkpoint_cb_best = ModelCheckpoint(save_top_k=1,
                                        monitor='val/score_epoch',
                                        filename='best-{epoch}-{step}',
                                        mode="max"
                                        )
        
        checkpoint_cb = ModelCheckpoint(save_last=True)

        lr_monitor = LearningRateMonitor(logging_interval="step", 
                                            log_momentum=False,
                                            log_weight_decay=False)
        
        if self.progress_bar:
            return [checkpoint_cb_best, checkpoint_cb, lr_monitor]
        else:
            progress_bar = LitProgressBar()
            return [checkpoint_cb_best, checkpoint_cb, progress_bar, lr_monitor]
