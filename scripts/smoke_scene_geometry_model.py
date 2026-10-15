"""Real DINO/LoRA forward-backward and checkpoint export for cross-runtime parity."""
import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from navsim.agents.drivoR.scene_geometry.geometry_loss import geometry_loss
from navsim.agents.drivoR.scene_geometry.target_builder import SceneGeometryTargetBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    base = OmegaConf.load(root / "navsim/planning/script/config/common/agent/drivoR.yaml")
    extra = OmegaConf.load(root / "navsim/planning/script/config/common/agent/drivoR_scene_geometry.yaml")
    cfg = OmegaConf.merge(base.config, extra.config)
    cfg.image_size = [224, 112]  # bounded integration check, not a training recipe
    cfg.image_backbone.model_weights = str(root / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors")
    cfg.scene_geometry.cache_root = str(Path(args.labels) / "cache")
    for key, value in dict(noc=10., dac=13., ddc=6., ttc=14., ep=15., comfort=2.).items():
        cfg[key] = value
    report = json.loads((Path(args.labels) / "report.json").read_text())
    sample = report["samples"][0]
    builder = SceneGeometryTargetBuilder(cfg)
    targets = builder.load_cached(sample["log"], sample["token"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2)
    agent = DrivoRAgent(cfg, lr_args={"name": "AdamW", "base_lr": 2e-4, "base_batch_size": 64},
                       scheduler_args=None, use_ray=False).to(device)
    model = agent._drivor_model
    optimizer = agent.get_optimizers()[0]
    grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in grouped for p in model.parameters() if p.requires_grad)
    features = {"image": torch.randn(1, 4, 3, 112, 224, device=device),
                "ego_status": torch.randn(1, 4, 11, device=device)}
    target = {key: value[None].to(device) for key, value in targets.items()}
    model.train()
    with torch.autocast(device, enabled=device == "cuda"):
        pred = model(features)
        loss = geometry_loss(pred["scene_geometry"], target)["geometry/loss"] + pred["proposals"].square().mean()
        loss = loss + sum(value.square().mean() for value in pred["pred_logit"].values())
    loss.backward()
    for name, parameter in model.named_parameters():
        if "scene_geometry" in name or "geometry_adapter" in name:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.scene_embeds.grad is not None and torch.isfinite(model.scene_embeds.grad).all()
    lora = [(n, p.grad) for n, p in model.named_parameters() if "linear_a_" in n or "linear_b_" in n]
    assert lora and any(g is not None and g.abs().sum() > 0 for _, g in lora)
    model.eval()
    with torch.no_grad():
        reference = model(features)
    torch.save({key: value.cpu() for key, value in features.items()}, out / "inputs.pt")
    torch.save({key: reference[key].cpu() for key in ("proposals", "pdm_score", "trajectory")}, out / "reference.pt")
    # The tunnel bridge caps individual writes at 16 MiB; shard ONLY this disposable fixture.
    shards, chunk, size = [], {}, 0
    for key, value in agent.state_dict().items():
        value = value.cpu()
        if chunk and size + value.numel() * value.element_size() > 8 * 1024 * 1024:
            name = f"checkpoint_{len(shards):03}.pt"
            torch.save(chunk, out / name); shards.append(name); chunk, size = {}, 0
        chunk["agent." + key] = value
        size += value.numel() * value.element_size()
    if chunk:
        name = f"checkpoint_{len(shards):03}.pt"
        torch.save(chunk, out / name); shards.append(name)
    (out / "checkpoint_shards.json").write_text(json.dumps(shards))
    OmegaConf.save(OmegaConf.create({"agent": {"config": cfg}}), out / "training_config.yaml")
    print("FULL_MODEL_AMP_OK", {"loss": float(loss.detach()), "lora_tensors": len(lora), "output": str(out)})


if __name__ == "__main__":
    main()
