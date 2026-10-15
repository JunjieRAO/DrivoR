"""Bounded real-data/AMP smoke check; does not start a training experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.drivoR.scene_geometry.target_builder import SceneGeometryTargetBuilder
from navsim.agents.drivoR.scene_geometry.geometry_encoder import SceneGeometryEncoder, GeometryAdapter
from navsim.agents.drivoR.scene_geometry.geometry_loss import geometry_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    base = OmegaConf.load("navsim/planning/script/config/common/agent/drivoR.yaml")
    extra = OmegaConf.load("navsim/planning/script/config/common/agent/drivoR_scene_geometry.yaml")
    cfg = OmegaConf.merge(base.config, extra.config)
    cfg.scene_geometry.cache_root = str(out / "cache")
    builder = SceneGeometryTargetBuilder(OmegaConf.create({"scene_geometry": cfg.scene_geometry}))
    data = Path(args.data_root)
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                               has_route=True, max_scenes=args.samples)
    loader = SceneLoader(data_path=data / "navsim_logs/trainval", sensor_blobs_path=data / "sensor_blobs/trainval",
                         scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = []
    for token in loader.tokens:
        scene = loader.get_scene_from_token(token)
        targets = builder.cache_scene(scene)
        loaded = builder.compute_targets(scene)
        assert all(torch.equal(value, loaded[key]) for key, value in targets.items())
        row = {key: int(value.sum()) for key, value in targets.items()
               if key.endswith(("valid", "overflow", "available", "unknown"))}
        row.update(token=token, log=scene.scene_metadata.log_name)
        stats.append(row)
        fig, ax = plt.subplots(figsize=(9, 6))
        roi = np.array(cfg.scene_geometry.roi)
        offset, scale = (roi[:2] + roi[2:]) / 2, (roi[2:] - roi[:2]) / 2
        for xy, cls, valid in zip(targets["sg_map_geometry"], targets["sg_map_class"], targets["sg_map_valid"]):
            if valid:
                points = xy.numpy().reshape(-1, 2) * scale + offset
                ax.plot(*points.T, color="black" if cls == 0 else "royalblue", linewidth=1)
                if cls == 1:
                    ax.annotate("", xy=points[-1], xytext=points[-3], arrowprops={"arrowstyle": "->", "color": "royalblue"})
        from matplotlib.patches import Polygon
        for geo, cls, valid in zip(targets["sg_agent_geometry"], targets["sg_agent_class"], targets["sg_agent_valid"]):
            if valid:
                x, y = geo[:2].numpy() * scale + offset
                length, width = geo[2:4].exp().numpy()
                yaw = np.arctan2(geo[4], geo[5])
                corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]
                rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                ax.add_patch(Polygon(corners @ rotation.T + [x, y], fill=False, edgecolor="crimson"))
        ax.scatter([0], [0], marker="^", color="green", label="Ego rear axle")
        ax.set(xlim=roi[[0, 2]], ylim=roi[[1, 3]], xlabel="Ego x (m)", ylabel="Ego y (m)", title=token)
        ax.set_aspect("equal"); ax.legend(); fig.tight_layout()
        fig.savefig(out / f"{token}.png"); plt.close(fig)
    if not stats:
        raise RuntimeError("No real scenes selected")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SceneGeometryEncoder(cfg, 64).to(device).train()
    adapter = GeometryAdapter(cfg).to(device).train()
    memory = torch.randn(1, 64, cfg.tf_d_model, device=device, requires_grad=True)
    target = {key: value[None].to(device) for key, value in targets.items()}
    query = torch.randn(1, 64, cfg.tf_d_model, device=device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(adapter.parameters()), lr=2e-4)
    # A bounded two-step test uses a conservative scale; production Lightning adapts its scale.
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda", init_scale=128)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, enabled=device == "cuda"):
            prediction = encoder(memory)
            proposals = adapter(query, prediction)
            loss = geometry_loss(prediction, target)["geometry/loss"] + proposals.square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in list(encoder.named_parameters()) + list(adapter.named_parameters()):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        scaler.step(optimizer); scaler.update()
        losses.append(float(loss.detach()))
    report = {"samples": stats, "device": device, "amp_losses": losses, "spec_hash": builder.spec_hash}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
