"""Cache labels with the exact training split/filter, without loading sensor pixels."""
import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataclasses import SensorConfig, SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.agents.drivoR.scene_geometry.target_builder import SceneGeometryTargetBuilder


_WORKER = None
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def empty_stats():
    return {"count": 0, "agent_overflow": 0, "map_overflow": 0, "missing_map": 0, "unknown_agents": 0}


def prepare_jobs(cfg):
    """Select globally before sharding so max_scenes retains its original split semantics."""
    jobs = []
    for split in ("train", "val"):
        scene_filter = instantiate(cfg.train_test_split.scene_filter)
        logs = cfg[split + "_logs"]
        scene_filter.log_names = ([name for name in scene_filter.log_names if name in logs]
                                 if scene_filter.log_names is not None else logs)
        loader = SceneLoader(data_path=Path(cfg.navsim_log_path),
                             sensor_blobs_path=Path(cfg.sensor_blobs_path),
                             scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors())
        for log_name, tokens in loader.get_tokens_list_per_log().items():
            jobs.append((split, log_name, tokens))
        # Workers reload just one log; never pickle the full raw dataset into each process.
        del loader
    return jobs


def initialize_worker(geometry_config, data_path, sensor_path, filter_fields, threads):
    global _WORKER
    for variable in _THREAD_VARIABLES:
        os.environ[variable] = str(threads)
    import torch
    torch.set_num_threads(threads)
    builder = SceneGeometryTargetBuilder(OmegaConf.create({"scene_geometry": geometry_config}))
    _WORKER = (builder, Path(data_path), Path(sensor_path), SceneFilter(**filter_fields))


def cache_log(job):
    """One task owns one log, reusing its map resources for all selected frames."""
    split, log_name, tokens = job
    builder, data_path, sensor_path, base_filter = _WORKER
    stats, loader = empty_stats(), None
    for token in tokens:
        try:
            path = builder.cache_path(log_name, token)
            if path.exists():
                targets = builder.load_cached(log_name, token)
            else:
                if loader is None:
                    loader = SceneLoader(data_path=data_path, sensor_blobs_path=sensor_path,
                                         scene_filter=replace(base_filter, log_names=[log_name],
                                                              tokens=tokens, max_scenes=None),
                                         sensor_config=SensorConfig.build_no_sensors())
                scene = loader.get_scene_from_token(token)
                if scene.scene_metadata.log_name != log_name:
                    raise ValueError("Selected log does not match scene metadata")
                targets = builder.cache_scene(scene)
            stats["count"] += 1
            stats["agent_overflow"] += int(targets["sg_agent_overflow"])
            stats["map_overflow"] += int(targets["sg_map_overflow"])
            stats["missing_map"] += int(not targets["sg_map_available"])
            stats["unknown_agents"] += int(targets["sg_agent_unknown"])
        except Exception as exc:
            raise RuntimeError(f"Geometry cache failed: split={split}, log={log_name}, token={token}") from exc
    return {**stats, "tokens": [[log_name, token] for token in tokens]}


def aggregate_results(jobs, results):
    splits = {split: {**empty_stats(), "tokens": []} for split in ("train", "val")}
    # Completion order must not change token ordering or the manifest identity.
    for (split, _, _), result in zip(jobs, results):
        for key in empty_stats():
            splits[split][key] += result[key]
        splits[split]["tokens"].extend(result["tokens"])
    return splits


def run_jobs(jobs, initializer_args, requested_workers, threads):
    if requested_workers < 1 or threads < 1:
        raise ValueError("cache_num_workers and cache_worker_threads must be positive integers")
    workers = min(requested_workers, len(jobs))
    print(f"Geometry caching: requested_workers={requested_workers}, active_workers={workers}, "
          f"threads_per_worker={threads}, log_tasks={len(jobs)}", flush=True)
    if not jobs:
        return [], workers
    if workers == 1:
        initialize_worker(*initializer_args)
        results = []
        for index, job in enumerate(jobs):
            results.append(cache_log(job))
            print(f"Completed logs: {index + 1}/{len(jobs)}", flush=True)
        return results, workers
    # Set limits before spawn imports NumPy/Torch, not only in the worker initializer.
    previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    for name in _THREAD_VARIABLES:
        os.environ[name] = str(threads)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker, initargs=initializer_args) as executor:
            futures = {executor.submit(cache_log, job): index for index, job in enumerate(jobs)}
            results = [None] * len(jobs)
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    results[futures[future]] = future.result()
                    print(f"Completed logs: {completed}/{len(jobs)}", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return results, workers
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@hydra.main(config_path="config/training", config_name="default_training", version_base=None)
def main(cfg):
    builder = SceneGeometryTargetBuilder(cfg.agent.config)
    if builder.root is None:
        raise ValueError("Set agent.config.scene_geometry.cache_root")
    geometry = cfg.agent.config.scene_geometry
    workers = int(geometry.get("cache_num_workers", 72))
    threads = int(geometry.get("cache_worker_threads", 1))
    if workers < 1 or threads < 1:
        raise ValueError("cache_num_workers and cache_worker_threads must be positive integers")
    jobs = prepare_jobs(cfg)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    fields = asdict(replace(scene_filter, log_names=None, tokens=None, max_scenes=None))
    args = (OmegaConf.to_container(geometry, resolve=True), str(cfg.navsim_log_path),
            str(cfg.sensor_blobs_path), fields, threads)
    results, active_workers = run_jobs(jobs, args, workers, threads)
    summary = {"spec": builder.spec, "spec_hash": builder.spec_hash,
               "execution": {"requested_workers": workers, "active_workers": active_workers,
                             "threads_per_worker": threads},
               "splits": aggregate_results(jobs, results)}
    builder.root.mkdir(parents=True, exist_ok=True)
    temporary = builder.root / f"manifest.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(summary, indent=2))
    # Per-run manifests cannot silently turn a pilot cache into a full-cache manifest.
    import hashlib
    identity = hashlib.sha256(json.dumps(summary["splits"], sort_keys=True).encode()).hexdigest()[:12]
    os.replace(temporary, builder.root / f"manifest_{identity}.json")
    print(json.dumps({s: {k: v for k, v in values.items() if k != "tokens"}
                      for s, values in summary["splits"].items()}, indent=2))
    print("Geometry cache:", builder.root)


if __name__ == "__main__":
    main()
