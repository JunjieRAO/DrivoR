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


if __name__ == "__main__":
    unittest.main()