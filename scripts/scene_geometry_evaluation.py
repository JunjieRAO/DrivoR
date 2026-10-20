"""正式 top1 two-stage 缓存检查、模型导出和评测；不使用训练几何 GT。"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


def integer(name, default, minimum=1):
    value = int(os.environ.get(name, default))
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}")
    return value


def quoted(key, value):
    # Hydra 的路径值须保留引号，尤其 checkpoint 文件名包含 '='。
    return f"{key}={json.dumps(str(value))}"


def select_checkpoint(repo):
    explicit = os.environ.get("CHECKPOINT_PATH")
    run = os.environ.get("TRAIN_RUN_DIR")
    if explicit:
        checkpoint = Path(explicit).expanduser().resolve()
    else:
        if not run:
            runs = [p for p in (repo / "exp/ke/drivor_sg_gen").glob("*")
                    if (p / "code/hydra/config.yaml").is_file()]
            if not runs:
                raise FileNotFoundError("未找到训练结果，请设置 TRAIN_RUN_DIR 或 CHECKPOINT_PATH + TRAIN_CONFIG")
            run = max(runs, key=lambda p: p.stat().st_mtime)
        candidates = list(Path(run).glob("lightning_logs/version_*/checkpoints/best*.ckpt"))
        if len(candidates) != 1:
            raise ValueError(f"{run} 中 best checkpoint 数量为 {len(candidates)}，请显式设置 CHECKPOINT_PATH")
        checkpoint = candidates[0].resolve()
    config = os.environ.get("TRAIN_CONFIG")
    if not config:
        config = next((p / "code/hydra/config.yaml" for p in checkpoint.parents
                       if (p / "code/hydra/config.yaml").is_file()), None)
    if not checkpoint.is_file() or not config or not Path(config).is_file():
        raise FileNotFoundError("checkpoint 或对应训练配置不存在；请设置 CHECKPOINT_PATH、TRAIN_CONFIG")
    return checkpoint, Path(config).resolve()


def execute(command, cwd):
    print("[RUN] " + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def compose_config(repo, kind, overrides):
    from hydra import compose, initialize_config_dir
    directory, name = (("metric_caching", "default_metric_caching") if kind == "cache"
                       else ("pdm_scoring", "default_run_pdm_score"))
    with initialize_config_dir(config_dir=str(repo / "navsim/planning/script/config" / directory), version_base=None):
        return compose(config_name=name, overrides=overrides)


def expected_tokens(cfg):
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    scene_loader = SceneLoader(
        synthetic_sensor_path=None, original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path), synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors())
    tokens = set(scene_loader.tokens)
    if not tokens:
        raise RuntimeError("评测场景为空，检查数据目录和 split")
    return tokens


def inspect_cache(cache, tokens):
    from navsim.common.dataloader import MetricCacheLoader
    metadata = list((cache / "metadata").glob("*.csv"))
    if not metadata:
        return {"expected": len(tokens), "missing": len(tokens), "missing_files": 0}
    # 当前官方 loader 只读取一个 CSV，多个索引不能假定自动合并。
    if len(metadata) != 1:
        raise RuntimeError(f"{cache}/metadata 有多个 CSV，当前 MetricCacheLoader 只读取第一个；请先整理索引")
    loader = MetricCacheLoader(cache)
    missing = tokens - set(loader.tokens)
    missing_files = sum(not Path(loader.metric_cache_paths[t]).is_file() for t in tokens - missing)
    report = {"expected": len(tokens), "indexed": len(loader.tokens),
              "missing": len(missing), "missing_files": missing_files}
    # 每次检查实际反序列化一个缓存，提前发现 NumPy 等环境兼容错误。
    if not missing and not missing_files:
        loader.get_from_token(min(tokens))
    return report


def verify_results(output, tokens):
    files = list(output.glob("*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"预期一个评测结果 CSV，实际 {len(files)} 个")
    with files[0].open() as stream:
        rows = list(csv.DictReader(stream))
    summary_names = {"extended_pdm_score_stage_one", "extended_pdm_score_stage_two", "extended_pdm_score_combined"}
    scenarios = [r for r in rows if r["token"] not in summary_names]
    summary = [r for r in rows if r["token"] in summary_names]
    if len(scenarios) != len(tokens) or {r["token"] for r in scenarios} != tokens:
        raise RuntimeError("结果场景覆盖不完整或含重复 token，请查看日志")
    if {r["token"] for r in summary} != summary_names:
        raise RuntimeError("结果缺少 two-stage 汇总行")
    if any(r["valid"].lower() != "true" or not math.isfinite(float(r["score"])) for r in rows):
        raise RuntimeError("结果包含失败场景、无效 pseudo-closed-loop 汇总或非有限分数，请查看 CSV")
    result = {r["token"]: float(r["score"]) for r in summary}
    print("[RESULT] " + json.dumps(result), flush=True)
    return {"csv": str(files[0]), "scores": result, "scenarios": len(scenarios)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache", "eval", "all"))
    args = parser.parse_args()
    repo = Path(os.environ["REPO_ROOT"]).resolve()
    evaluation = Path(os.environ["EVAL_REPO_ROOT"]).resolve()
    exp = Path(os.environ["NAVSIM_EXP_ROOT"])
    split = os.environ.get("TRAIN_TEST_SPLIT", "navhard_two_stage")
    if split != "navhard_two_stage":
        raise ValueError("本脚本固定正式 navhard_two_stage；其他 split 需要单独核对映射和数据路径")
    cache = Path(os.environ.get("METRIC_CACHE_PATH", str(exp / "navhard_two_stage_metric_cache"))).resolve()
    workers = integer("CPU_WORKERS", 72)
    dry = os.environ.get("DRY_RUN", "0") == "1"
    check = os.environ.get("CHECK_ONLY", "0") == "1"
    base = [f"train_test_split={split}", quoted("metric_cache_path", cache), f"worker.threads_per_node={workers}"]
    cache_overrides = base + ["force_feature_computation=false"]
    cache_command = [sys.executable, evaluation / "navsim/planning/script/run_metric_caching.py", *cache_overrides]
    export_command = eval_command = None
    if args.stage != "cache":
        checkpoint, training_config = select_checkpoint(repo)
        from export_scene_geometry import load_training_agent_config
        mode = load_training_agent_config(training_config).scene_geometry.mode
        if mode not in ("off", "aux", "gen", "gen_scorer"):
            raise ValueError(f"不支持的训练 mode: {mode}")
        experiment = f"drivor_sg_{mode}_top1_two_stage"
        stamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")
        output = Path(os.environ.get("OUTPUT_DIR", str(exp / experiment / stamp))).resolve()
        if output.exists():
            raise FileExistsError(f"请使用全新的 OUTPUT_DIR，避免混淆结果: {output}")
        export_command = [sys.executable, repo / "scripts/export_scene_geometry.py", "--destination", evaluation,
                          "--mode", mode, "--training-config", training_config]
        devices = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",") if g.strip()]
        if not devices:
            raise ValueError("GPU_IDS 不能为空")
        overrides = base + ["agent=drivoR_scene_geometry", quoted("agent.checkpoint_path", checkpoint),
            f"experiment_name={experiment}", quoted("output_dir", output), "gpu_inference=true", "gpu_device=cuda:0",
            f"gpu_num_devices={len(devices)}", f"gpu_batch_size={integer('GPU_BATCH_SIZE', 16)}",
            f"gpu_num_workers={integer('GPU_NUM_WORKERS', 8, 0)}",
            f"max_scenarios_per_task={integer('MAX_SCENARIOS_PER_TASK', 32)}"]
        eval_command = [sys.executable, evaluation / "navsim/planning/script/run_pdm_score.py", *overrides]
        print(f"[INFO] checkpoint: {checkpoint}\n[INFO] training config: {training_config}\n[INFO] mode: {mode}\n[INFO] output: {output}", flush=True)
    print(f"[INFO] Python: {sys.executable}\n[INFO] metric cache: {cache}", flush=True)
    if dry:
        if args.stage in ("cache", "all"):
            print("[DRY RUN] 仅当完整性检查发现缺失时执行: " + shlex.join(map(str, cache_command)))
        for command in (export_command, eval_command):
            if command:
                print("[DRY RUN] " + shlex.join(map(str, command)))
        return
    # 串行保护共用的导出模块和 metric cache，避免另一个任务在本次运行中修改它们。
    import fcntl
    lock_path = exp / "scene_geometry_pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("已有 scene geometry pipeline 运行中；请等待完成") from None
        cfg = compose_config(evaluation, "cache", cache_overrides)
        for path in (cfg.navsim_log_path, cfg.synthetic_scenes_path, os.environ["NUPLAN_MAPS_ROOT"]):
            if not Path(path).is_dir():
                raise FileNotFoundError(path)
        tokens = expected_tokens(cfg)
        report = inspect_cache(cache, tokens)
        print("[CACHE] " + json.dumps(report), flush=True)
        if report["missing"] or report["missing_files"]:
            if args.stage == "eval" or check:
                raise RuntimeError("metric cache 不完整，请运行 nav2_cache_two_stage.sh 或 nav2_eval_scene_geometry_all.sh")
            execute(cache_command, evaluation)
            report = inspect_cache(cache, tokens)
            if report["missing"] or report["missing_files"]:
                raise RuntimeError(f"缓存生成后仍不完整: {report}")
        else:
            print("[CACHE] 已完整，复用现有 metric cache。", flush=True)
        if args.stage == "cache":
            return
        execute(export_command, evaluation)
        eval_cfg = compose_config(evaluation, "eval", overrides)
        for path in (eval_cfg.original_sensor_path, eval_cfg.synthetic_sensor_path):
            if not Path(path).is_dir():
                raise FileNotFoundError(path)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < len(devices):
            raise RuntimeError("可见 CUDA 设备不足，请设置 GPU_IDS")
        if check:
            from hydra.utils import instantiate
            agent = instantiate(eval_cfg.agent)
            agent.initialize()
            print("[CHECK PASS] metric cache 完整、配置有效、checkpoint 严格加载成功；未执行完整评测。", flush=True)
            return
        with checkpoint.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        export_manifest = json.loads((evaluation / "navsim/agents/drivor_scene_geometry/export_manifest.json").read_text())
        manifest = {"status": "running", "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "training_config": str(training_config), "mode": mode, "cache": str(cache), "cache_check": report,
            "export_code_hash": export_manifest["code_hash"], "command": list(map(str, eval_command)),
            "python": sys.executable, "torch": torch.__version__, "expected_scenarios": len(tokens)}
        output.mkdir(parents=True, exist_ok=False)
        manifest_path = output / "pipeline_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        try:
            execute(eval_command, evaluation)
            manifest.update(verify_results(output, tokens), status="complete")
        except Exception as exc:
            manifest.update(status="failed", error=str(exc))
            raise
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
