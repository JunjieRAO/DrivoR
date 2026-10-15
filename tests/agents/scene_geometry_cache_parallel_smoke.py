"""Real-data serial/spawn parity check on two train and two validation scenes."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.planning.script import run_scene_geometry_caching as caching


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = Path(args.output)
    with initialize_config_dir(config_dir=str(root / "navsim/planning/script/config/training"), version_base=None):
        cfg = compose(config_name="default_training", overrides=[
            "agent=drivoR_scene_geometry", "train_test_split=navtrain",
            "++train_test_split.scene_filter.max_scenes=2"])
    assert cfg.agent.config.scene_geometry.cache_num_workers == 72
    jobs = caching.prepare_jobs(cfg)
    assert len(jobs) == 2 and sum(len(job[2]) for job in jobs) == 4
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    fields = asdict(replace(scene_filter, log_names=None, tokens=None, max_scenes=None))
    results, builders = [], []
    for mode, workers in (("serial", 1), ("parallel", 72)):
        geometry = OmegaConf.to_container(cfg.agent.config.scene_geometry, resolve=True)
        geometry["cache_root"] = str(out / mode)
        initializer_args = (geometry, str(cfg.navsim_log_path), str(cfg.sensor_blobs_path), fields, 1)
        result, active = caching.run_jobs(jobs, initializer_args, workers, 1)
        assert active == min(workers, len(jobs))
        results.append(caching.aggregate_results(jobs, result))
        builders.append(caching.SceneGeometryTargetBuilder(OmegaConf.create({"scene_geometry": geometry})))
    assert results[0] == results[1]
    assert builders[0].spec_hash == builders[1].spec_hash
    for _, log, tokens in jobs:
        for token in tokens:
            serial, parallel = [b.load_cached(log, token) for b in builders]
            assert serial.keys() == parallel.keys()
            assert all(torch.equal(serial[key], parallel[key]) for key in serial)
    before = {str(p): p.stat().st_mtime_ns for p in builders[1].root.rglob("*.gz")}
    repeated, _ = caching.run_jobs(jobs, initializer_args, 72, 1)
    assert caching.aggregate_results(jobs, repeated) == results[1]
    assert before == {str(p): p.stat().st_mtime_ns for p in builders[1].root.rglob("*.gz")}
    report = {"status": "SERIAL_PARALLEL_CACHE_OK", "requested_workers": 72,
              "active_workers_for_two_logs": 2, "tensor_equal": True, "resume_preserved_files": True,
              "spec_hash": builders[0].spec_hash, "splits": results[0]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
