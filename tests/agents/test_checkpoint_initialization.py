import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from navsim.agents.drivoR.drivor_agent import DrivoRAgent


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_backbone = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 2))
        self.scene_embeds = nn.Parameter(torch.randn(1, 2))
        self.hist_encoding = nn.Linear(3, 2)
        self.init_feature = nn.Embedding(2, 2)
        self.trajectory_decoder = nn.Linear(2, 2)
        self.traj_head = nn.Linear(2, 3)
        self.pos_embed = nn.Linear(3, 2)
        self.scorer_attention = nn.Linear(2, 2)
        self.scorer = nn.Linear(2, 1)


class CheckpointInitializationTest(unittest.TestCase):
    def check_loading(self, perception_only):
        model = ToyModel()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        saved = {f"agent._drivor_model.{key}": value + 5 for key, value in original.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save({"state_dict": saved}, path)
            with patch("navsim.agents.drivoR.drivor_agent.DrivoRModel", return_value=model):
                agent = DrivoRAgent({}, {}, checkpoint_path=str(path), checkpoint_perception_only=perception_only)
            agent.initialize()
        for key, value in model.state_dict().items():
            inherited = not perception_only or key.startswith("image_backbone.") or key == "scene_embeds"
            torch.testing.assert_close(value, original[key] + (5 if inherited else 0))

    def test_only_perception_inherited(self):
        self.check_loading(True)

    def test_full_checkpoint_loading_unchanged(self):
        self.check_loading(False)

    def test_missing_perception_rejected(self):
        model = ToyModel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save({"state_dict": {}}, path)
            with patch("navsim.agents.drivoR.drivor_agent.DrivoRModel", return_value=model):
                agent = DrivoRAgent({}, {}, checkpoint_path=str(path), checkpoint_perception_only=True)
            with self.assertRaisesRegex(ValueError, "Incomplete perception"):
                agent.initialize()


class LaneKeepingCheckpointTest(unittest.TestCase):
    def make_model(self):
        model = ToyModel()
        model.scorer = nn.ModuleDict({
            "pred_score": nn.ModuleDict({
                "comfort": nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)),
                "lane_keeping": nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)),
            }),
        })
        return model

    def load(self, model, state, lk):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save({"state_dict": {f"agent._drivor_model.{key}": value for key, value in state.items()}}, path)
            with patch("navsim.agents.drivoR.drivor_agent.DrivoRModel", return_value=model):
                agent = DrivoRAgent({"lk": lk}, {}, checkpoint_path=str(path))
            agent.initialize()
        return agent

    def test_new_head_checkpoint_loads_strictly(self):
        model = self.make_model()
        saved = {key: value + 5 for key, value in model.state_dict().items()}
        self.load(model, saved, lk=2.0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[key])

    def test_legacy_checkpoint_initializes_only_lk_with_ranking_disabled(self):
        model = self.make_model()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        saved = {key: value + 5 for key, value in original.items()
                 if not key.startswith("scorer.pred_score.lane_keeping.")}
        with self.assertWarnsRegex(UserWarning, "LK ranking disabled"):
            self.load(model, saved, lk=0.0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved.get(key, original[key]))

    def test_legacy_checkpoint_rejects_untrained_lk_ranking(self):
        model = self.make_model()
        saved = {key: value for key, value in model.state_dict().items()
                 if not key.startswith("scorer.pred_score.lane_keeping.")}
        with self.assertRaisesRegex(ValueError, "agent.config.lk=0"):
            self.load(model, saved, lk=2.0)

    def test_partial_lk_or_other_missing_keys_remain_strict(self):
        for missing in ("scorer.pred_score.lane_keeping.0.weight", "hist_encoding.weight"):
            with self.subTest(missing=missing):
                model = self.make_model()
                saved = dict(model.state_dict())
                del saved[missing]
                with self.assertRaises(RuntimeError):
                    self.load(model, saved, lk=0.0)


if __name__ == "__main__":
    unittest.main()