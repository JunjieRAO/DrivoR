from types import SimpleNamespace
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from navsim.planning.script import run_scene_geometry_caching as caching


def test_worker_defaults_and_empty_selection():
    cfg = OmegaConf.load("navsim/planning/script/config/common/agent/drivoR_scene_geometry.yaml")
    assert cfg.config.scene_geometry.cache_num_workers == 72
    assert cfg.config.scene_geometry.cache_worker_threads == 1
    assert caching.run_jobs([], (), 72, 1) == ([], 0)
    with pytest.raises(ValueError):
        caching.run_jobs([], (), 0, 1)


def test_selected_split_and_global_limit_preserved(monkeypatch):
    seen = []
    class Loader:
        def __init__(self, **kwargs):
            selected = kwargs["scene_filter"]
            seen.append(selected)
        def get_tokens_list_per_log(self):
            return {seen[-1].log_names[0]: ["a", "b"]}
    monkeypatch.setattr(caching, "SceneLoader", Loader)
    cfg = OmegaConf.create({
        "train_test_split": {"scene_filter": {"_target_": "navsim.common.dataclasses.SceneFilter",
                                               "log_names": ["train_log", "val_log", "excluded"],
                                               "max_scenes": 2}},
        "train_logs": ["train_log"], "val_logs": ["val_log"],
        "navsim_log_path": "/logs", "sensor_blobs_path": "/sensors"})
    assert caching.prepare_jobs(cfg) == [("train", "train_log", ["a", "b"]), ("val", "val_log", ["a", "b"])]
    assert [s.max_scenes for s in seen] == [2, 2]


def test_existing_cache_skips_scene_loading_and_reports_failure(tmp_path, monkeypatch):
    file = tmp_path / "cached.gz"
    file.touch()
    class Builder:
        def cache_path(self, log, token):
            return file
        def load_cached(self, log, token):
            return {"sg_agent_overflow": 3, "sg_map_overflow": 2,
                    "sg_map_available": True, "sg_agent_unknown": 0}
    builder = Builder()
    monkeypatch.setattr(caching, "_WORKER", (builder, None, None, None))
    def forbid_loader(**kwargs):
        raise AssertionError("Cached logs should not load raw scenes or maps")
    monkeypatch.setattr(caching, "SceneLoader", forbid_loader)
    result = caching.cache_log(("train", "log", ["a", "b"]))
    assert result["count"] == 2 and result["agent_overflow"] == 6
    assert result["tokens"] == [["log", "a"], ["log", "b"]]
    def corrupt(*args):
        raise ValueError("bad cache")
    monkeypatch.setattr(builder, "load_cached", corrupt)
    with pytest.raises(RuntimeError, match="split=train, log=log, token=a"):
        caching.cache_log(("train", "log", ["a"]))


def test_aggregation_keeps_selection_order():
    jobs = [("train", "log_b", ["b"]), ("train", "log_a", ["a"]), ("val", "log_v", ["v"])]
    results = [{**caching.empty_stats(), "count": 1, "tokens": [[log, tokens[0]]]}
               for _, log, tokens in jobs]
    combined = caching.aggregate_results(jobs, results)
    assert combined["train"]["tokens"] == [["log_b", "b"], ["log_a", "a"]]
    assert combined["train"]["count"] == 2
