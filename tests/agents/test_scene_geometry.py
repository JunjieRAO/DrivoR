from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from shapely.geometry import Polygon, LineString
from navsim.agents.drivoR.scene_geometry.geometry_encoder import SceneGeometryEncoder, GeometryAdapter
from navsim.agents.drivoR.scene_geometry.geometry_targets import (
    boundary_lines, vectorize_lines, build_geometry_targets, local_geometry)
from navsim.agents.drivoR.scene_geometry.geometry_loss import geometry_loss, pair_cost
from navsim.agents.drivoR.scene_geometry.target_builder import SceneGeometryTargetBuilder


def config():
    cfg = OmegaConf.load("navsim/planning/script/config/common/agent/drivoR.yaml").config
    cfg.scene_geometry = OmegaConf.load("navsim/planning/script/config/common/agent/drivoR_scene_geometry.yaml").config.scene_geometry
    cfg.tf_d_model, cfg.tf_d_ffn, cfg.num_scene_tokens = 32, 64, 2
    cfg.scene_geometry.num_agent_queries = 8
    cfg.scene_geometry.num_map_queries = 8
    cfg.scene_geometry.map_points = 4
    cfg.scene_geometry.cache_mode = "on_the_fly"
    cfg.scene_geometry.cache_root = None
    return cfg


def scene(empty=False):
    boxes = np.empty((0, 7)) if empty else np.array([[5, 2, 0, 4, 2, 1, 0], [2, 1, 0, .5, .5, 1, 1]])
    names = [] if empty else ["vehicle", "traffic_cone"]
    frame = SimpleNamespace(annotations=SimpleNamespace(boxes=boxes, names=names),
                            ego_status=SimpleNamespace(ego_pose=np.zeros(3)))
    return SimpleNamespace(frames=[frame], map_api=None,
                           scene_metadata=SimpleNamespace(num_history_frames=1, log_name="log", initial_token="token"))


def batch_targets(cfg, empty=False):
    return {k: v[None] for k, v in build_geometry_targets(scene(empty), cfg.scene_geometry).items()}


def test_boundary_union_hole_and_no_crop_edge():
    a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(2, 2), (3, 2), (3, 3), (2, 3)]])
    b = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
    lines = boundary_lines([a, b])
    assert sum(x.length for x in lines) == 64  # union perimeter 60 plus hole 4
    assert not vectorize_lines(lines, [4, 4, 6, 6], 4, 20)
    assert len(vectorize_lines(lines, [1, 1, 4, 4], 4, 20)) == 1


def test_direction_and_coordinate_transform():
    cfg = config()
    gt = torch.tensor([[0., 0., 1, 0, 2, 0, 3, 0]])
    reverse = gt.reshape(1, 4, 2).flip(1).flatten(1)
    boundary, _ = pair_cost(reverse, gt, torch.ones_like(gt), "map", torch.tensor([0]))
    center, _ = pair_cost(reverse, gt, torch.ones_like(gt), "map", torch.tensor([1]))
    assert boundary.item() == 0 and center.item() > 0
    transformed = local_geometry(LineString([(10, 21), (10, 22)]), [10, 20, np.pi / 2])
    assert np.allclose(np.asarray(transformed.coords), [[1, 0], [2, 0]])
    target = build_geometry_targets(scene(), cfg.scene_geometry)
    assert target["sg_agent_valid"].sum() == 2
    assert not target["sg_agent_mask"][0, 4:].any()  # nearest object is cone
    assert not target["sg_map_available"]


@pytest.mark.parametrize("empty", [False, True])
def test_loss_gradients_empty_and_padding(empty):
    torch.manual_seed(2)
    cfg = config()
    encoder = SceneGeometryEncoder(cfg, 8)
    memory = torch.randn(1, 8, 32, requires_grad=True)
    pred = encoder(memory)
    target = batch_targets(cfg, empty)
    result = geometry_loss(pred, target)
    assert torch.isfinite(result["geometry/loss"])
    result["geometry/loss"].backward()
    assert memory.grad is not None and torch.isfinite(memory.grad).all()
    assert all(p.grad is not None for p in encoder.parameters())
    # Padded garbage must not participate in matching or regression.
    target["sg_agent_geometry"][~target["sg_agent_valid"]] = 10000
    assert torch.allclose(geometry_loss(pred, target)["geometry/loss"], result["geometry/loss"])


def test_zero_gate_and_detached_geometry():
    cfg = config()
    adapter = GeometryAdapter(cfg).eval()
    pred = {}
    for kind, count, size, classes in (("agent", 8, 6, 4), ("map", 8, 8, 2)):
        pred[kind] = [{"tokens": torch.randn(1, count, 32, requires_grad=True),
                       "geometry": torch.randn(1, count, size, requires_grad=True),
                       "logits": torch.randn(1, count, classes, requires_grad=True)}]
    query = torch.randn(1, 64, 32)
    assert torch.equal(adapter(query, pred), query)
    adapter.agent.gate.data.fill_(.1)
    adapter.map.gate.data.fill_(.1)
    adapter(query, pred).square().mean().backward()
    for kind in pred:
        assert pred[kind][0]["tokens"].grad is not None
        assert pred[kind][0]["geometry"].grad is None
        assert pred[kind][0]["logits"].grad is None


def test_cache_strict_identity_and_spec(tmp_path):
    cfg = config()
    cfg.scene_geometry.cache_root = str(tmp_path)
    cfg.scene_geometry.cache_mode = "strict"
    builder = SceneGeometryTargetBuilder(cfg)
    with pytest.raises(FileNotFoundError):
        builder.compute_targets(scene())
    target = builder.cache_scene(scene())
    assert torch.equal(builder.compute_targets(scene())["sg_agent_geometry"], target["sg_agent_geometry"])
    cfg.scene_geometry.roi[0] = -30
    assert SceneGeometryTargetBuilder(cfg).spec_hash != builder.spec_hash


class TinyBackbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_features = cfg.tf_d_model
        self.projection = torch.nn.Linear(3, cfg.tf_d_model)
    def forward(self, image, tokens):
        return tokens.flatten(1, 2) + self.projection(image)[:, None]


def test_model_off_aux_gen_compatibility_and_scorer(monkeypatch):
    import navsim.agents.drivoR.drivor_model as module
    monkeypatch.setattr(module, "ImgEncoder", TinyBackbone)
    cfg = config()
    models = []
    for mode in ("off", "aux", "gen"):
        cfg.scene_geometry.mode = mode
        torch.manual_seed(2)
        models.append(module.DrivoRModel(OmegaConf.create(OmegaConf.to_container(cfg))).eval())
    baseline = models[0].state_dict()
    for model in models[1:]:
        state = model.state_dict()
        assert all(torch.equal(state[key], value) for key, value in baseline.items())
    assert all("geometry" not in name for name in baseline)
    features = {"image": torch.randn(1, 3), "ego_status": torch.randn(1, 4, 11)}
    outputs = [model(features) for model in models]
    for output in outputs:
        assert output["proposals"].shape == (1, 64, 8, 3)
        assert torch.equal(output["proposals"], outputs[0]["proposals"])
        assert torch.equal(output["pdm_score"], outputs[0]["pdm_score"])
    models[0].load_state_dict(baseline, strict=True)
    with pytest.raises(RuntimeError):
        models[2].load_state_dict(baseline, strict=True)
    models[2].load_state_dict(models[2].state_dict(), strict=True)
    assert not any("geometry" in key for key in models[2].scorer_attention.state_dict())


def test_agent_optimizer_ramp_and_cacheonly(tmp_path, monkeypatch):
    import navsim.agents.drivoR.drivor_model as module
    from navsim.agents.drivoR.drivor_agent import DrivoRAgent
    from navsim.planning.training.dataset import CacheOnlyDataset
    monkeypatch.setattr(module, "ImgEncoder", TinyBackbone)
    cfg = config()
    agent = DrivoRAgent(cfg, {"name": "AdamW", "base_lr": .0002, "base_batch_size": 64},
                       use_ray=False)
    optimizer = agent.get_optimizers()[0]
    parameters = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in parameters for p in agent.parameters() if p.requires_grad)
    agent.set_geometry_progress(0, 100)
    assert agent.geometry_ramp == 0
    agent.set_geometry_progress(5, 100)
    assert agent.geometry_ramp == .5
    agent.set_geometry_progress(20, 100)
    assert agent.geometry_ramp == 1
    cfg.scene_geometry.cache_root = str(tmp_path / "geometry")
    cfg.scene_geometry.cache_mode = "strict"
    builder = SceneGeometryTargetBuilder(cfg)
    root = tmp_path / "old"
    (root / "log/token").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        CacheOnlyDataset(str(root), [], [builder])
    builder.cache_scene(scene())
    dataset = CacheOnlyDataset(str(root), [], [builder])
    assert dataset[0][1]["sg_agent_valid"].sum() == 2


def test_loss_integration_and_custom_validation(monkeypatch):
    import navsim.agents.drivoR.drivor_model as module
    from navsim.agents.drivoR.drivor_agent import DrivoRAgent
    from navsim.planning.training.agent_lightning_module import AgentLightningModule
    monkeypatch.setattr(module, "ImgEncoder", TinyBackbone)
    cfg = config()
    class PlanningLoss(torch.nn.Module):
        def forward(self, targets, pred, config, scorer):
            return {"loss": pred["proposals"].square().mean()}
    agent = DrivoRAgent(cfg, {"name": "AdamW", "base_lr": .0002, "base_batch_size": 64},
                       loss=PlanningLoss(), use_ray=False).eval()
    features = {"image": torch.randn(1, 3), "ego_status": torch.randn(1, 4, 11)}
    targets = batch_targets(cfg)
    predictions = agent(features)
    agent.set_geometry_progress(5, 100)
    losses = agent.compute_loss(features, targets, predictions)
    assert torch.allclose(losses["loss"], predictions["proposals"].square().mean() + .5 * losses["geometry/loss"])
    def scores(targets, proposals):
        values = proposals.new_ones(proposals.shape[:2])
        return values.mean(), values.mean(), values, values.mean(), proposals.new_ones((len(proposals), 6))
    monkeypatch.setattr(agent, "compute_score", scores)
    lightning = AgentLightningModule(agent)
    logged = {}
    monkeypatch.setattr(lightning, "log", lambda key, value, **kwargs: logged.update({key: value}))
    lightning.validation_step((features, targets), 0)
    assert "val/geometry/loss" in logged
    assert "val/score" in logged
