"""Exercise actual spawn workers, not a mocked executor."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from navsim.offline_search import run as runner
ROOT = Path(__file__).resolve().parents[2]


def invoke(out, *extra):
    return subprocess.run([sys.executable, "-m", "navsim.offline_search.run", "demo",
        "--output", str(out), "--population", "4", "--generations", "1", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_spawn_parallel_matches_serial(tmp_path):
    outputs = [tmp_path / "serial", tmp_path / "parallel"]
    for out, workers in zip(outputs, [1, 2]):
        proc = invoke(out, "--workers", str(workers), "--scene-count", "2")
        assert proc.returncode == 0, proc.stdout + proc.stderr
    summaries = [json.loads((p / "summary.json").read_text()) for p in outputs]
    assert [s["execution"]["effective_workers"] for s in summaries] == [1, 2]
    assert [r["token"] for r in summaries[0]["scenes"]] == [r["token"] for r in summaries[1]["scenes"]]
    for row in summaries[0]["scenes"]:
        dirs = [p / row["folder"] for p in outputs]
        assert (dirs[0] / "candidate_archive.jsonl").read_bytes() == (dirs[1] / "candidate_archive.jsonl").read_bytes()
        scene_summaries = [json.loads((d / "summary.json").read_text()) for d in dirs]
        assert scene_summaries[0]["trace"] == scene_summaries[1]["trace"]
        arrays = [np.load(d / "engineering_candidates.npz") for d in dirs]
        assert arrays[0].files == arrays[1].files
        for key in arrays[0].files:
            np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
    assert summaries[0]["scenes"][0]["qualified_count"] >= 0


def test_workers_validation_clamp_and_no_overwrite(tmp_path):
    out = tmp_path / "out"
    assert invoke(out, "--workers", "0").returncode != 0
    assert not out.exists()
    assert invoke(out, "--workers", "4").returncode == 0
    assert json.loads((out / "execution.json").read_text())["effective_workers"] == 1
    before = (out / "summary.json").read_bytes()
    assert invoke(out).returncode != 0
    assert (out / "summary.json").read_bytes() == before


def test_pool_startup_failure_is_reported(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("pool unavailable")
    monkeypatch.setattr(runner, "ProcessPoolExecutor", fail)
    args = SimpleNamespace(command="demo", output=tmp_path / "failed", seed=2,
                           population=4, generations=1, workers=2, worker_threads=1, scene_count=2)
    assert runner.run(args) == 2
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["failed"] == summary["processed"] == 2
    assert all(r["search_status"] == "worker_process_error" for r in summary["scenes"])
