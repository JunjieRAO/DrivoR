from types import SimpleNamespace

import numpy as np
import torch

from navsim.agents.drivoR.drivor_features import DrivoRTargetBuilder


class _Config(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


def _frame(pose, boxes=None, names=None, velocities=None):
    boxes = np.zeros((0, 7), dtype=np.float32) if boxes is None else boxes
    names = [] if names is None else names
    velocities = (
        np.zeros((0, 3), dtype=np.float32) if velocities is None else velocities
    )
    return SimpleNamespace(
        ego_status=SimpleNamespace(ego_pose=np.asarray(pose, dtype=np.float64)),
        annotations=SimpleNamespace(
            boxes=boxes,
            names=names,
            velocity_3d=velocities,
        ),
    )


def test_future_targets_transform_frame_local_boxes_to_current_ego() -> None:
    config = _Config(
        num_poses=1,
        future_num_steps=1,
        future_occupancy_height=64,
        future_occupancy_width=64,
        future_occupancy_blur_sigma=1.0,
        lidar_min_x=-16.0,
        lidar_max_x=16.0,
        lidar_min_y=-16.0,
        lidar_max_y=16.0,
    )
    builder = DrivoRTargetBuilder(config)
    box = np.array([[2.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], dtype=np.float32)
    velocity = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
    scene = SimpleNamespace(
        scene_metadata=SimpleNamespace(num_history_frames=1),
        frames=[
            _frame([0.0, 0.0, 0.0]),
            _frame(
                [4.0, 0.0, np.pi / 2],
                boxes=box,
                names=["vehicle"],
                velocities=velocity,
            ),
        ],
    )

    targets = builder._compute_future_dynamic_targets(scene)

    occupancy = targets["future_dynamic_occupancy"]
    velocity_grid = targets["future_dynamic_velocity"]
    assert occupancy.shape == (1, 3, 64, 64)
    assert occupancy[0, 0].max() == 1
    assert occupancy[0, 1:].sum() == 0
    occupied = occupancy[0, 0] > 0.5
    rows, columns = torch.where(occupied)
    assert rows.float().mean() > 31.5
    assert columns.float().mean() > 31.5
    assert torch.allclose(
        velocity_grid[0, 0][occupied].mean(), torch.tensor(0.0), atol=1e-5
    )
    assert torch.allclose(
        velocity_grid[0, 1][occupied].mean(), torch.tensor(3.0), atol=1e-5
    )
    assert torch.all(targets["future_dynamic_valid"] == 1)


def test_future_targets_keep_empty_frames_valid() -> None:
    config = _Config(
        num_poses=1,
        future_num_steps=1,
        future_occupancy_height=32,
        future_occupancy_width=32,
        future_occupancy_blur_sigma=0.0,
        lidar_min_x=-10.0,
        lidar_max_x=10.0,
        lidar_min_y=-10.0,
        lidar_max_y=10.0,
    )
    builder = DrivoRTargetBuilder(config)
    scene = SimpleNamespace(
        scene_metadata=SimpleNamespace(num_history_frames=1),
        frames=[_frame([0.0, 0.0, 0.0]), _frame([0.0, 0.0, 0.0])],
    )

    targets = builder._compute_future_dynamic_targets(scene)

    assert targets["future_dynamic_occupancy"].sum() == 0
    assert targets["future_dynamic_velocity"].sum() == 0
    assert targets["future_dynamic_valid"].sum() == 32 * 32