from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
import torch
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from omegaconf import OmegaConf
from shapely.geometry import LineString, box

from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from navsim.agents.drivoR.structured_input.encoder import StructuredSceneEncoder
from navsim.agents.drivoR.structured_input.features import StructuredSceneFeatureBuilder, MAP_TYPE_QUOTAS
from navsim.agents.drivoR.transformer_decoder import Attention
from navsim.common.dataclasses import Scene
from navsim.planning.training.dataset import Dataset, CacheOnlyDataset


def config(enabled=True, fusion="both"):
    root = Path(__file__).resolve().parents[2]
    options = OmegaConf.load(root / "navsim/planning/script/config/common/agent/drivoR.yaml").config
    options.tf_d_model = 32
    options.tf_d_ffn = 64
    options.num_scene_tokens = 2
    options.proposal_num = 4
    options.ref_num = 1
    options.scorer_ref_num = 1
    options.structured_input = {"enabled": enabled, "fusion": fusion, "map_points": 8}
    return options


def structured_features(batch=2):
    return {
        "structured_agents": torch.randn(batch, 5, 12),
        "structured_agent_types": torch.zeros(batch, 5, dtype=torch.long),
        "structured_agent_valid": torch.ones(batch, 5, dtype=torch.bool),
        "structured_map_points": torch.randn(batch, 6, 8, 2),
        "structured_map_types": torch.arange(6)[None].repeat(batch, 1),
        "structured_map_valid": torch.ones(batch, 6, dtype=torch.bool),
        "structured_lights": torch.randn(batch, 3, 3),
        "structured_light_valid": torch.ones(batch, 3, dtype=torch.bool),
    }


class TinyBackbone(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        self.num_features = options.tf_d_model
        self.projection = torch.nn.Linear(3, options.tf_d_model)

    def forward(self, image, tokens):
        return tokens.flatten(1, 2) + self.projection(image)[:, None]


def model(options):
    with patch("navsim.agents.drivoR.drivor_model.ImgEncoder", TinyBackbone):
        return DrivoRModel(options).eval()


def test_padding_attention_ignores_masked_values_and_excludes_softmax_mass():
    attention = Attention(dim=32, num_heads=4).eval()
    queries = torch.randn(2, 3, 32)
    memory = torch.randn(2, 7, 32)
    padding = torch.tensor([[False, False, False, True, True, True, True]]).repeat(2, 1)
    expected = attention(queries, memory[:, :3])
    torch.testing.assert_close(attention(queries, memory, padding), expected)
    memory[:, 3:] = 10000.0
    torch.testing.assert_close(attention(queries, memory, padding), expected)


def test_entity_tokens_are_permutation_equivariant_and_mask_padding():
    encoder = StructuredSceneEncoder(config())
    features = structured_features()
    expected, valid = encoder(features)
    assert expected.shape == (2, 14, 32)
    assert valid.shape == (2, 14)
    start = 0
    for prefix, fields in (
        ("agent", ("structured_agents", "structured_agent_types")),
        ("map", ("structured_map_points", "structured_map_types")),
        ("light", ("structured_lights",)),
    ):
        valid_key = f"structured_{prefix}_valid"
        features[valid_key][:, -1] = False
        reference, reference_valid = encoder(features)
        for field in fields:
            features[field][:, -1] = float("nan") if features[field].is_floating_point() else 999
        actual, actual_valid = encoder(features)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(actual_valid, reference_valid)
        permutation = torch.randperm(features[valid_key].shape[1])
        for field in (*fields, valid_key):
            features[field] = features[field][:, permutation]
        end = start + len(permutation)
        actual, actual_valid = encoder(features)
        torch.testing.assert_close(actual[:, start:end], reference[:, start:end][:, permutation])
        torch.testing.assert_close(actual_valid[:, start:end], reference_valid[:, start:end][:, permutation])
        torch.testing.assert_close(actual[:, :start], reference[:, :start])
        torch.testing.assert_close(actual[:, end:], reference[:, end:])
        start = end


def test_empty_modalities_have_finite_zero_tokens_and_valid_backward():
    encoder = StructuredSceneEncoder(config())
    features = structured_features()
    for name in ("agent", "map", "light"):
        features[f"structured_{name}_valid"].fill_(False)
    output, valid = encoder(features)
    assert torch.equal(output, torch.zeros_like(output))
    assert not valid.any()
    output.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in encoder.parameters())


@pytest.mark.parametrize("fusion", ["gen", "scorer", "both"])
def test_tokens_reach_selected_decoders_and_receive_gradients(fusion):
    network = model(config(fusion=fusion))
    features = structured_features()
    features.update(image=torch.randn(2, 3), ego_status=torch.randn(2, 4, 11))
    memory_sizes = {}
    handles = []
    for name, decoder in (("gen", network.trajectory_decoder), ("scorer", network.scorer_attention)):
        def capture(module, arguments, name=name):
            memory_sizes[name] = arguments[1].shape[1]
        handles.append(decoder.register_forward_pre_hook(capture))
    output = network(features)
    visual_count = network.num_cams * network._config.num_scene_tokens
    assert memory_sizes["gen"] == visual_count + (14 if fusion in ("gen", "both") else 0)
    assert memory_sizes["scorer"] == visual_count + (14 if fusion in ("scorer", "both") else 0)
    for handle in handles:
        handle.remove()
    if fusion in ("gen", "both"):
        output["proposals"].square().mean().backward(retain_graph=True)
        assert network.structured_encoder.encoders["agents"][0].weight.grad.abs().sum() > 0
    network.zero_grad(set_to_none=True)
    output = network(features)
    sum(value.square().mean() for value in output["pred_logit"].values()).backward()
    if fusion in ("scorer", "both"):
        assert network.structured_encoder.encoders["map"][0].weight.grad.abs().sum() > 0


def test_polyline_encoder_preserves_sample_order():
    encoder = StructuredSceneEncoder(config())
    features = structured_features()
    expected, _ = encoder(features)
    features["structured_map_points"] = features["structured_map_points"].flip(-2)
    actual, _ = encoder(features)
    torch.testing.assert_close(actual[:, :5], expected[:, :5])
    assert not torch.allclose(actual[:, 5:11], expected[:, 5:11])


def test_default_capacities_encode_one_token_per_entity_and_not_per_type():
    options = config()
    features = StructuredSceneFeatureBuilder(options).compute_features(scene())
    assert features["structured_agents"].shape == (128, 12)
    assert features["structured_map_points"].shape == (256, 8, 2)
    assert features["structured_lights"].shape == (16, 3)
    encoder = StructuredSceneEncoder(options)
    batch = {name: value[None] for name, value in features.items()}
    tokens, valid = encoder(batch)
    assert tokens.shape == (1, 400, 32)
    assert valid.shape == (1, 400)
    torch.testing.assert_close(valid[:, :128], batch["structured_agent_valid"])
    torch.testing.assert_close(valid[:, 128:384], batch["structured_map_valid"])
    torch.testing.assert_close(valid[:, 384:], batch["structured_light_valid"])
    assert (tokens[~valid] == 0).all()
    batch["structured_agents"][:, 0, 0] += 10.0
    changed, _ = encoder(batch)
    assert not torch.allclose(changed[:, 0], tokens[:, 0])
    torch.testing.assert_close(changed[:, 1:], tokens[:, 1:])


@pytest.mark.parametrize("fusion", ["gen", "scorer", "both"])
def test_full_model_excludes_empty_structured_slots_from_attention(fusion):
    network = model(config(fusion=fusion))
    features = structured_features()
    features.update(image=torch.randn(2, 3), ego_status=torch.randn(2, 4, 11))
    for name in ("agent", "map", "light"):
        features[f"structured_{name}_valid"].fill_(False)
    expected = network(features)
    for name in tuple(features):
        if name.startswith("structured_"):
            features[name] = features[name][:, :0]
    actual = network(features)
    torch.testing.assert_close(actual["proposals"], expected["proposals"])
    torch.testing.assert_close(actual["pdm_score"], expected["pdm_score"])


def test_disabled_path_and_original_initialization_are_unchanged():
    torch.manual_seed(2)
    original_options = config(False)
    del original_options.structured_input
    baseline = model(original_options)
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(2)
    disabled = model(config(False))
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    assert baseline.state_dict().keys() == disabled.state_dict().keys()
    torch.manual_seed(2)
    enabled = model(config(True))
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(enabled.state_dict()[name], value, rtol=0, atol=0)
    features = {"image": torch.randn(2, 3), "ego_status": torch.randn(2, 4, 11)}
    for name, value in baseline(features).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(disabled(features)[name], value, rtol=0, atol=0)


class CurrentOnlyFrames:
    def __init__(self, previous, current):
        self.history = [previous, current]

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < 2:
            raise AssertionError("Future frame access is forbidden")
        return self.history[index]


class TinyMap:
    def __init__(self):
        lane = SimpleNamespace(
            id="lane", baseline_path=SimpleNamespace(linestring=LineString([(10, 20), (10, 30)])),
            left_boundary=SimpleNamespace(linestring=LineString([(8, 20), (8, 30)])),
            right_boundary=SimpleNamespace(linestring=LineString([(12, 20), (12, 30)])),
        )
        self.connector = SimpleNamespace(**{**vars(lane), "id": "connector"})
        self.objects = {
            SemanticMapLayer.LANE: [lane], SemanticMapLayer.LANE_CONNECTOR: [self.connector],
            SemanticMapLayer.STOP_LINE: [SimpleNamespace(polygon=box(9, 20, 11, 20.2))],
            SemanticMapLayer.CROSSWALK: [], SemanticMapLayer.WALKWAYS: [],
            SemanticMapLayer.INTERSECTION: [],
        }

    def get_available_map_objects(self):
        return list(self.objects)

    def get_proximal_map_objects(self, position, radius, layers):
        return {layer: self.objects[layer] for layer in layers}


def scene():
    bounds = np.array([[1., 2., 0., 4., 2., 1., np.pi / 4]])
    previous = SimpleNamespace(
        timestamp=1000000, ego_status=SimpleNamespace(ego_pose=np.array([10., 20., 0.])),
        annotations=SimpleNamespace(boxes=bounds.copy(), names=["vehicle"],
                                    track_tokens=["actor"], velocity_3d=np.array([[2., 0., 0.]])),
    )
    current = SimpleNamespace(
        timestamp=1500000, ego_status=SimpleNamespace(ego_pose=np.array([10., 20., np.pi / 2])),
        annotations=SimpleNamespace(boxes=bounds.copy(), names=["vehicle"],
                                    track_tokens=["actor"], velocity_3d=np.array([[0., -4., 0.]])),
        traffic_lights=[("connector", True)],
    )
    return SimpleNamespace(scene_metadata=SimpleNamespace(num_history_frames=2),
                           frames=CurrentOnlyFrames(previous, current), map_api=TinyMap())


def test_current_features_coordinates_sizes_causal_acceleration_and_map_types():
    builder = StructuredSceneFeatureBuilder(config())
    features = builder.compute_features(scene())
    np.testing.assert_allclose(features["structured_agents"][0],
                               [1, 2, np.sin(np.pi / 4), np.cos(np.pi / 4), 2, 4,
                                0, -4, 0, -4, 1, 1], atol=1e-6)
    assert features["structured_agent_valid"].sum() == 1
    assert set(features["structured_map_types"][features["structured_map_valid"]].tolist()) == set(range(7))
    assert features["structured_map_points"][features["structured_map_valid"]].isfinite().all()
    torch.testing.assert_close(features["structured_lights"][0], torch.tensor([0., 0., 1.]))
    assert features["structured_light_valid"].sum() == 1


def test_missing_agent_kinematics_are_flagged_not_invented():
    current_scene = scene()
    current_scene.frames[1].annotations.velocity_3d[:] = np.nan
    features = StructuredSceneFeatureBuilder(config()).compute_features(current_scene)
    torch.testing.assert_close(features["structured_agents"][0, 6:], torch.zeros(6))
    current_scene.frames[1].annotations.velocity_3d[:] = [1., 0., 0.]
    current_scene.frames[1].annotations.track_tokens = ["new_actor"]
    features = StructuredSceneFeatureBuilder(config()).compute_features(current_scene)
    torch.testing.assert_close(features["structured_agents"][0, 8:], torch.tensor([0., 0., 1., 0.]))


def populated_scene(positions, velocities=None, ego_velocity=(0., 0.)):
    current_scene = scene()
    current = current_scene.frames[1]
    current.ego_status.ego_pose = np.zeros(3)
    current.ego_status.ego_velocity = np.asarray(ego_velocity)
    count = len(positions)
    bounds = np.zeros((count, 7))
    bounds[:, :2] = positions
    bounds[:, 3:6] = [4., 2., 1.]
    current.annotations = SimpleNamespace(
        boxes=bounds, names=["vehicle"] * count,
        track_tokens=[f"track_{index:03d}" for index in range(count)],
        velocity_3d=np.c_[np.zeros((count, 2)) if velocities is None else velocities, np.zeros(count)],
    )
    return current_scene


@pytest.mark.parametrize("capacity", [64, 128])
def test_agent_capacity_keeps_approaching_risk_beyond_nearest_and_is_deterministic(capacity):
    positions = [(5 + index * .1, 15.) for index in range(capacity + 6)] + [(50., 0.)]
    current_scene = populated_scene(positions, ego_velocity=(10., 0.))
    options = config()
    options.structured_input.max_agents = capacity
    options.structured_input.use_map = False
    options.structured_input.use_traffic_lights = False
    builder = StructuredSceneFeatureBuilder(options)
    selected = builder.compute_features(current_scene)
    assert selected["structured_agent_valid"].sum() == capacity
    assert (selected["structured_agents"][:, :2] == torch.tensor([50., 0.])).all(dim=1).any()
    assert len(torch.unique(selected["structured_agents"][:, :2], dim=0)) == capacity
    annotations = current_scene.frames[1].annotations
    annotations.boxes = annotations.boxes[::-1].copy()
    annotations.velocity_3d = annotations.velocity_3d[::-1].copy()
    annotations.names = annotations.names[::-1]
    annotations.track_tokens = annotations.track_tokens[::-1]
    torch.testing.assert_close(builder.compute_features(current_scene)["structured_agents"], selected["structured_agents"])
    options.structured_input.agent_selection = "nearest"
    nearest = StructuredSceneFeatureBuilder(options).compute_features(current_scene)
    assert not (nearest["structured_agents"][:, :2] == torch.tensor([50., 0.])).all(dim=1).any()


def test_agent_spatial_coverage_preserves_other_sectors():
    positions = [(5 + index * .05, 4.) for index in range(100)] + [(-15., -15.), (-15., 15.), (10., -20.)]
    options = config()
    options.structured_input.max_agents = 64
    options.structured_input.use_map = False
    options.structured_input.use_traffic_lights = False
    features = StructuredSceneFeatureBuilder(options).compute_features(populated_scene(positions))
    for position in positions[-3:]:
        assert (features["structured_agents"][:, :2] == torch.tensor(position)).all(dim=1).any()


def map_candidate(category, source, distance):
    return {"category": category, "source": source, "distance": distance,
            "priority": (distance,), "points": np.array([[distance, 0.], [distance + 1., 0.]])}


@pytest.mark.parametrize("capacity", [64, 256])
def test_map_soft_quotas_reserve_all_types_and_fill_missing_types(capacity):
    options = config()
    options.structured_input.max_map_lines = capacity
    builder = StructuredSceneFeatureBuilder(options)
    candidates = [map_candidate(category, f"source_{index}", index + category * 100.)
                  for category in range(10) for index in range(70)]
    selected = builder._select_map(candidates)
    assert len(selected) == capacity
    expected_quotas = (np.asarray(MAP_TYPE_QUOTAS) * capacity // 64).tolist()
    assert [sum(row["category"] == category for row in selected) for category in range(10)] == expected_quotas
    only_boundaries = [map_candidate(2, f"source_{index}", float(index)) for index in range(capacity + 26)]
    selected = builder._select_map(only_boundaries)
    assert len(selected) == capacity
    assert len({row["source"] for row in selected}) == capacity
    assert {row["source"] for row in selected} == {f"source_{index}" for index in range(capacity)}


def test_map_source_diversity_precedes_extra_segments_and_quotas_scale():
    options = config()
    options.structured_input.max_map_lines = 8
    builder = StructuredSceneFeatureBuilder(options)
    candidates = [map_candidate(2, "long_boundary", index * .1) for index in range(40)]
    candidates.extend(map_candidate(2, f"other_{index}", 10. + index) for index in range(6))
    selected = builder._select_map(candidates)
    assert len(selected) == 8
    assert sum(row["source"] == "long_boundary" for row in selected) == 2
    assert builder._map_quotas().sum() == 8
    current_scene = scene()
    current_scene.map_api.objects[SemanticMapLayer.LANE] *= 3
    duplicate_features = StructuredSceneFeatureBuilder(config()).compute_features(current_scene)
    original_features = StructuredSceneFeatureBuilder(config()).compute_features(scene())
    torch.testing.assert_close(duplicate_features["structured_map_points"], original_features["structured_map_points"])


def test_map_and_light_selection_have_deterministic_ties():
    builder = StructuredSceneFeatureBuilder(config())
    candidates = [map_candidate(category, f"source_{index}", float(index))
                  for category in range(10) for index in range(10)]
    expected = builder._select_map(candidates)
    actual = builder._select_map(candidates[::-1])
    assert [(row["category"], row["source"]) for row in actual] == [
        (row["category"], row["source"]) for row in expected]
    current_scene = scene()
    current_scene.frames[1].traffic_lights *= 3
    repeated = builder.compute_features(current_scene)
    assert repeated["structured_light_valid"].sum() == 1


def test_lights_keep_sixteen_relevant_connectors_and_deduplicate_statuses():
    current_scene = scene()
    current = current_scene.frames[1]
    current.ego_status.ego_pose = np.zeros(3)
    current_scene.map_api.objects[SemanticMapLayer.LANE] = [SimpleNamespace(
        id="ego_lane", baseline_path=SimpleNamespace(linestring=LineString([(0., 0.), (60., 0.)])),
    )]
    positions = [(15. + index, 1.) for index in range(20)] + [(2. + index, 20.) for index in range(20)]
    connectors = [SimpleNamespace(id=f"connector_{index}", baseline_path=SimpleNamespace(
        linestring=LineString([position, (position[0] + 5., position[1])]),
    )) for index, position in enumerate(positions)]
    current_scene.map_api.objects[SemanticMapLayer.LANE_CONNECTOR] = connectors
    current.traffic_lights = [(item.id, False) for item in connectors] + [(connectors[0].id, True)]
    options = config()
    options.structured_input.use_map = False
    features = StructuredSceneFeatureBuilder(options).compute_features(current_scene)
    assert features["structured_light_valid"].sum() == 16
    assert features["structured_lights"].shape == (16, 3)
    assert (features["structured_lights"][:, 1] == 1).all()
    torch.testing.assert_close(features["structured_lights"][0], torch.tensor([15., 1., 1.]))


def test_feature_cache_identity_and_modality_ablation():
    options = config()
    builder = StructuredSceneFeatureBuilder(options)
    options.structured_input.fusion = "scorer"
    assert StructuredSceneFeatureBuilder(options).get_unique_name() == builder.get_unique_name()
    assert builder.get_unique_name().startswith("structured_current_v2_")
    options.structured_input.map_selection = "nearest"
    assert StructuredSceneFeatureBuilder(options).get_unique_name() != builder.get_unique_name()
    options.structured_input.map_selection = "balanced"
    options.structured_input.max_agents = 2
    assert StructuredSceneFeatureBuilder(options).get_unique_name() != builder.get_unique_name()
    options.structured_input.use_map = False
    options.structured_input.use_traffic_lights = False
    current_scene = scene()
    current_scene.map_api = None
    features = StructuredSceneFeatureBuilder(options).compute_features(current_scene)
    assert features["structured_agent_valid"].sum() == 1
    assert not features["structured_map_valid"].any()
    with pytest.raises(ValueError, match="Scene"):
        builder.compute_features(SimpleNamespace())
    with pytest.raises(ValueError, match="requires the current scene map"):
        builder.compute_features(current_scene)


class SensorBuilder:
    def get_unique_name(self):
        return "sensor_test"

    def compute_features(self, sensor_input):
        assert sensor_input == "sensor_input"
        return {"image": torch.ones(3), "ego_status": torch.zeros(4, 11)}


class TargetBuilder:
    def get_unique_name(self):
        return "target_test"

    def compute_targets(self, current_scene):
        return {"trajectory": torch.zeros(8, 3)}


def test_dataset_dispatch_and_cache_only_round_trip(tmp_path):
    current_scene = scene()
    current_scene.scene_metadata.log_name = "log"
    current_scene.scene_metadata.initial_token = "token"
    current_scene.get_agent_input = lambda: "sensor_input"
    loader = SimpleNamespace(
        tokens=["token"], get_scene_from_token=lambda token: current_scene,
        get_agent_input_from_token=lambda token: "sensor_input",
    )
    builders = [SensorBuilder(), StructuredSceneFeatureBuilder(config())]
    targets = [TargetBuilder()]
    online = Dataset(loader, builders, targets)[0][0]
    cached = Dataset(loader, builders, targets, cache_path=str(tmp_path))[0][0]
    cache_only = CacheOnlyDataset(str(tmp_path), builders, targets)[0][0]
    for name in builders[1].compute_features(current_scene):
        torch.testing.assert_close(online[name], cached[name])
        torch.testing.assert_close(online[name], cache_only[name])


def test_parallel_cache_entry_dispatches_current_scene(tmp_path):
    from navsim.planning.training.dataset_cache import Dataset as CacheDataset

    current_scene = scene()
    current_scene.scene_metadata.log_name = "log"
    current_scene.scene_metadata.initial_token = "token"
    current_scene.get_agent_input = lambda: "sensor_input"
    loader = SimpleNamespace(tokens=["token"], get_scene_from_token=lambda token: current_scene)
    builders = [SensorBuilder(), StructuredSceneFeatureBuilder(config())]
    CacheDataset(loader, builders, [TargetBuilder()], cache_path=str(tmp_path))
    features = CacheOnlyDataset(str(tmp_path), builders, [TargetBuilder()])[0][0]
    assert features["structured_agent_valid"].sum() == 1
    assert features["structured_light_valid"].sum() == 1


def test_agent_marks_gt_privilege_and_inference_receives_scene():
    options = config()
    with patch("navsim.agents.drivoR.drivor_model.ImgEncoder", TinyBackbone):
        agent = DrivoRAgent(options, {"name": "AdamW", "base_lr": 0.0002, "base_batch_size": 64},
                            use_ray=False).eval()
    assert agent.requires_scene
    assert len(agent.get_feature_builders()) == 2
    with pytest.raises(ValueError, match="Structured GT inference requires Scene"):
        agent.compute_trajectory(SimpleNamespace())
    current_scene = scene()
    native_scene = Scene(current_scene.scene_metadata, current_scene.map_api, current_scene.frames)
    native_scene.get_agent_input = lambda: "sensor_input"
    with patch.object(agent, "get_feature_builders", return_value=[SensorBuilder(),
                      StructuredSceneFeatureBuilder(options)]):
        trajectory = agent.compute_trajectory(native_scene)
    assert trajectory.poses.shape == (8, 3)
    assert np.isfinite(trajectory.poses).all()
    optimizer = agent.get_optimizers()[0]
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert all(id(parameter) in optimized for parameter in agent._drivor_model.structured_encoder.parameters())