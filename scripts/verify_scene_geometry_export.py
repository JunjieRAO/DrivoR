"""Run from the evaluation repository with ONLY its own navsim on PYTHONPATH."""
import argparse
import json
from pathlib import Path
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    args = parser.parse_args()
    fixture = Path(args.fixture)
    from navsim.agents.drivor_scene_geometry.drivor_agent import DrivoRSceneGeometryAgent
    import navsim
    cfg = OmegaConf.load(fixture / "training_config.yaml").agent.config
    sampling = instantiate(cfg.trajectory_sampling)
    agent = DrivoRSceneGeometryAgent(cfg, str(fixture / "checkpoint.pt"), sampling)
    # Read every tensor from real torch files, then exercise initialize's strict key mapping.
    # Only the smoke fixture is sharded to respect the bridge's per-file write limit.
    from unittest.mock import patch
    state = {}
    for name in json.loads((fixture / "checkpoint_shards.json").read_text()):
        shard = torch.load(fixture / name, map_location="cpu", weights_only=True)
        if state.keys() & shard.keys():
            raise ValueError("Duplicate checkpoint shard keys")
        state.update(shard)
    with patch("torch.load", return_value={"state_dict": state}):
        agent.initialize()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent.to(device).eval()
    inputs = {key: value.to(device) for key, value in torch.load(fixture / "inputs.pt", weights_only=True).items()}
    reference = torch.load(fixture / "reference.pt", weights_only=True)
    with torch.no_grad():
        prediction = agent(inputs)
    errors = {}
    for key, expected in reference.items():
        actual = prediction[key].cpu()
        errors[key] = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert agent.get_target_builders() == []
    assert len(agent.get_feature_builders()) == 1
    agent.get_sensor_config()
    report = {"status": "EXPORT_PARITY_OK", "navsim_path": list(navsim.__path__), "max_abs_error": errors}
    (fixture / "export_parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
