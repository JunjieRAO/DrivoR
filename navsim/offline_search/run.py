"""CLI for manifest construction, synthetic UI checks, and real V1 engineering runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
import hashlib
import html
import importlib.metadata
import json
import lzma
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
from .core import Config, Candidate, candidate_id, residual, search, representatives, quality, stable_seed
from .report import write_report


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@lru_cache(maxsize=64)
def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    versions = {}
    for name in ["numpy", "scipy", "shapely", "pandas", "nuplan-devkit"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else "unavailable"
    return {"python": sys.version, "platform": platform.platform(), "versions": versions,
            "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "module_sha256": {p.name: fingerprint(str(p)) for p in Path(__file__).parent.glob("*.py")}}


def read_allowlist(path):
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".json":
        values = json.loads(text)
    elif path.suffix in {".yaml", ".yml"}:
        import yaml
        values = yaml.safe_load(text)
    else:
        values = [x.strip() for x in text.splitlines() if x.strip()]
    if isinstance(values, dict):
        values = values.get("tokens")
    if not isinstance(values, list) or not values or not all(isinstance(x, str) for x in values):
        raise ValueError("allowlist must contain string tokens, as a list or {tokens: [...]} (no numeric tokens)")
    return set(values)


def prepare(args):
    allowed = read_allowlist(args.train_tokens)
    excluded = set()
    for path in args.exclude_tokens:
        excluded |= read_allowlist(path)
    overlap = allowed & excluded
    if overlap:
        raise ValueError(f"Training allowlist overlaps excluded validation/test tokens: {len(overlap)}")
    caches = {}
    for path in args.cache_root.rglob("metric_cache.pkl"):
        token = path.parent.name
        if token in caches:
            raise ValueError(f"ambiguous duplicate cache token: {token}")
        caches[token] = str(path.resolve())
    groups = defaultdict(list)
    for path in sorted(args.scene_root.glob("*.pkl")):
        with path.open("rb") as f:
            frames = pickle.load(f)
        # Search by token, not default SceneFilter.frame_interval (which can omit requested tokens).
        for i in range(3, len(frames) - 10):
            token = frames[i]["token"]
            if token in allowed:
                log = frames[i]["log_name"]
                groups[log].append({"token": token, "log_name": log, "source_log": str(path.resolve()),
                    "frame_start": i - 3, "metric_cache": caches.get(token), "declared_split": "train",
                    "allowlist_sha256": fingerprint(str(args.train_tokens.resolve()))})
    selected = []
    logs = sorted(groups, key=lambda log: stable_seed(args.seed, log))
    for round_index in range(2):
        for log in logs:
            choices = sorted(groups[log], key=lambda r: stable_seed(args.seed, r["token"]))
            if round_index < len(choices) and len(selected) < args.count:
                selected.append(choices[round_index])
    if not selected:
        raise ValueError("No train allowlist tokens with current + 10 future frames found")
    if len({r['token'] for r in selected}) != len(selected):
        raise ValueError("duplicate scene tokens in raw logs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(args.output), "requested": args.count, "selected": len(selected),
                      "logs": len(set(r["log_name"] for r in selected)),
                      "missing_cache": sum(r["metric_cache"] is None for r in selected)}))


class DemoEvaluator:
    """Synthetic smoke fixture, NOT a NAVSIM simulator or a PDMS estimator."""
    def __init__(self, config):
        from shapely.geometry import box, LineString
        from .geometry import Actor, NominalValidator
        self.config = config
        self.ego_local = np.array([[3.8, 1.], [-1., 1.], [-1., -1.], [3.8, -1.]])
        self.road = box(-10, -4, 55, 4)
        self.raw_path = np.array([[0., 0., 0.], [50., 0., 0.]])
        # One adjacent moving actor and one static object deliberately expose bad candidates.
        poses = np.c_[18 + np.arange(41) * .04, np.full(41, 2.8), np.zeros(41)]
        self.actors = [Actor("demo_adjacent", poses, np.array([[2, .7], [-2, .7], [-2, -.7], [2, -.7]]))]
        self.validator = NominalValidator(config, self.ego_local, self.actors, self.road,
            LineString(self.raw_path[:, :2]), 3., speed_limit_fn=lambda _: 13.9)
        self.evaluations = 0
        self.gt = self.evaluate_theta(np.zeros(10))
        self.gt.island = "GT"
        self.roundtrip = self.evaluate_theta(np.r_[np.full(5, .005), np.zeros(5)])
        self.roundtrip.island = "demo_reference_variant"
        self.replay_error = None

    def evaluate_theta(self, theta):
        controls = residual(theta)
        states = np.zeros((41, 11))
        states[0, 3] = 6.
        for k in range(40):
            prev = states[k]
            acc = prev[5] + (controls[k, 0] - prev[5]) / 3
            steer = prev[7] + controls[k, 1] * .1
            states[k + 1] = prev
            states[k + 1, :2] += prev[3] * .1 * np.array([np.cos(prev[2]), np.sin(prev[2])])
            states[k + 1, 2] += prev[3] * np.tan(prev[7]) / 3 * .1
            states[k + 1, 3] += acc * .1
            states[k + 1, 5] = acc
            states[k + 1, 7:9] = [steer, controls[k, 1]]
        score = float(np.clip(states[-1, 0] / 32, 0, 1))
        metrics = dict(no_at_fault_collisions=1., drivable_area_compliance=1., ego_progress=score,
                       time_to_collision_within_bound=1., comfort=1., driving_direction_compliance=1., score=score)
        gate = self.validator.check(states, metrics)
        poses = states[5::5, :3].copy()
        self.evaluations += 1
        return Candidate(candidate_id(poses), poses, states.copy(), states.copy(), metrics, gate,
                         generated=states.copy(), controls=controls, theta=theta.copy())

    def seeds(self):
        return {"B0": np.zeros(10), "B1": np.r_[[.6, .6, 0, 0, 0], np.zeros(5)],
                "B3": np.r_[[-.6, -.6, 0, 0, 0], np.zeros(5)],
                "B6": np.r_[np.zeros(5), [.015, .015, -.015, -.015, 0]],
                "B7": np.r_[np.zeros(5), [-.015, -.015, .015, .015, 0]]}, [{"demo_only": True}]


def run_scene(evaluator, token, out, config, metadata, demo):
    out.mkdir()
    start = time.perf_counter()
    if evaluator.gt.score + config.min_gain > 1:
        seeds, seed_notes = {}, [{"status": "gt_upper_bound"}]
    elif getattr(evaluator, "unknown", []):
        seeds, seed_notes = {}, [{"status": "structural_cache_error", "reasons": evaluator.unknown}]
    else:
        seeds, seed_notes = evaluator.seeds()
    # Stream metadata even if the process is interrupted; completed arrays saved below.
    with (out / "candidate_archive.jsonl").open("w", encoding="utf-8") as f:
        def record(c):
            f.write(json.dumps(c.record(), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
        record(evaluator.gt)
        record(evaluator.roundtrip)
        all_c, archive, trace = search(config, token, evaluator.gt, seeds, evaluator.evaluate_theta, record)
    selected = representatives(archive, evaluator.gt)
    # Re-evaluate saved poses for real data. Demo only implements its synthetic controls.
    if not demo:
        for c in archive:
            again = evaluator.evaluate_poses(c.poses)
            if (again.id != c.id or not again.passed
                    or any(abs(again.metrics[k] - c.metrics[k]) > 1e-9 for k in c.metrics)
                    or not np.allclose(again.executed, c.executed, atol=1e-8, rtol=0)):
                raise ValueError("Final saved-pose re-evaluation mismatch; invalidate scene")
    everything = {c.id: c for c in [evaluator.gt, evaluator.roundtrip] + all_c}
    arrays = {}
    for c in everything.values():
        for key in ["poses", "reference", "executed", "generated", "controls", "theta"]:
            value = getattr(c, key)
            if value is not None:
                arrays[c.id + "_" + key] = value
    np.savez_compressed(out / "engineering_candidates.npz", **arrays)
    failure_counts = Counter(reason for c in all_c for reason in c.gate["reasons"])
    search_status = ("gt_upper_bound" if evaluator.gt.score + config.min_gain > 1 else
                     "precheck_blocked" if getattr(evaluator, "unknown", []) else
                     "search_not_started" if not all_c else
                     "nominal_improvement_found" if archive else
                     "no_verifiable_candidate_in_budget" if not any(c.gate.get('data_verifiable') for c in all_c) else
                     "no_feasible_candidate_in_budget" if not any(c.passed for c in all_c) else
                     "no_nominal_improvement_in_budget")
    diagnostics = {'time_alignment': getattr(evaluator, 'time_diagnostics', None),
                   'actors': evaluator.validator.lifecycle_diagnostics,
                   'structural_cache_errors': getattr(evaluator, 'unknown', []),
                   'map_limit_samples_gt': evaluator.gt.gate.get('speed_limit_diagnostics'),
                   'scope': 'conditional engineering replay; not certified teacher data'}
    dump(out / 'diagnostics.json', diagnostics)
    # Partial-track poses are saved as NaN plus an explicit mask in NPZ, never as invented states.
    actor_arrays = {}
    for i, actor in enumerate(evaluator.actors):
        actor_arrays[f'{i}_poses'] = actor.poses
        actor_arrays[f'{i}_observed'] = actor.valid
        actor_arrays[f'{i}_local_vertices'] = actor.local
    np.savez_compressed(out / 'actor_replay.npz', **actor_arrays)
    summary = {"token": token, "status": "engineering_completed", "search_status": search_status, "demo": demo,
               "schema_version": 2, "search_started": bool(all_c),
               "search_generation_count": sum('generation' in row for row in trace),
               "diagnostics_file": "diagnostics.json",
               "score_kind": "synthetic_proxy" if demo else "official_v1_pdms",
               "gt": evaluator.gt.record(), "roundtrip": evaluator.roundtrip.record(),
               "gt_command_replay_max_error": evaluator.replay_error,
               "roundtrip_score_delta": evaluator.roundtrip.score - evaluator.gt.score,
               "search_unique_candidates": len(all_c), "formal_calls_including_init_and_recheck": evaluator.evaluations,
               "search_unique_control_evaluations": max((r.get('evaluations', 0) for r in trace), default=0),
               "candidate_verifiable_count": sum(c.gate.get('data_verifiable', False) for c in all_c),
               "failure_category_counts": dict(Counter(category for c in all_c for category in {reason.split(':')[0] for reason in c.gate['reasons']})),
               "checked_nominal_pass_count": sum(c.passed for c in all_c),
               "qualified_count": len({c.id for c in all_c if c.passed and c.score >= evaluator.gt.score + config.min_gain}),
               "nominal_improvement_archive": [c.id for c in archive],
               "engineering_diverse_representatives": [c.id for c in selected],
               "failure_counts": dict(failure_counts),
               "seconds": time.perf_counter() - start + metadata.get("setup_seconds", 0),
               "setup_seconds": metadata.get("setup_seconds", 0),
               "strict_improvement_count": None, "export_certified": False,
               "training_export_enabled": False, "seeds": seed_notes, "trace": trace}
    dump(out / "summary.json", summary)
    dump(out / "config.json", asdict(config))
    # Keep every improving archive member, plus diagnostically different failures.
    visual = {c.id: c for c in [evaluator.roundtrip] + archive}
    failures = sorted([c for c in all_c if not c.passed], key=quality)
    seen = set()
    for c in failures:
        kind = tuple(c.gate["reasons"])
        if kind not in seen:
            visual.setdefault(c.id, c)
            seen.add(kind)
    for c in sorted(all_c, key=quality):
        if len(visual) >= 60:
            break
        visual.setdefault(c.id, c)
    write_report(out / "report.html", token, list(visual.values())[:60], evaluator.gt,
                 evaluator.road, evaluator.actors, evaluator.ego_local, evaluator.raw_path[:, :2], trace,
                 {**metadata, "summary": summary, "config": asdict(config)}, demo)
    return summary


def run(args):
    config = Config(seed=args.seed, population=args.population, generations=args.generations)
    args.output.mkdir(parents=True, exist_ok=False)
    env = environment()
    dump(args.output / "environment.json", env)
    dump(args.output / "config.json", asdict(config))
    demo = args.command == "demo"
    if demo:
        rows = [{"token": "SYNTHETIC_DEMO", "declared_split": "synthetic"}]
    else:
        rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not rows or len(rows) > 16:
            raise ValueError("Engineering manifest must contain 1..16 rows")
        if len({r['token'] for r in rows}) != len(rows):
            raise ValueError("duplicate manifest tokens")
        if any(r.get("declared_split") != "train" or not isinstance(r.get("token"), str) for r in rows):
            raise ValueError("Only explicit train manifests with string tokens are allowed")
        if any(count > 2 for count in Counter(r["log_name"] for r in rows).values()):
            raise ValueError("Engineering stage allows at most two scenes per log")
        if args.map_root:
            os.environ["NUPLAN_MAPS_ROOT"] = str(args.map_root.resolve())
        dump(args.output / "manifest.json", rows)
    results = []
    for index, row in enumerate(rows):
        setup_start = time.perf_counter()
        token = row["token"]
        # Token is data, not a path segment.
        folder = f"{index:02d}_" + hashlib.sha256(token.encode()).hexdigest()[:12]
        print(f"[{index + 1}/{len(rows)}] {token}", flush=True)
        try:
            if demo:
                evaluator = DemoEvaluator(config)
                meta = {"environment": env}
            else:
                from navsim.common.dataclasses import Scene, SensorConfig
                from .official import OfficialEvaluator
                if not row.get("metric_cache"):
                    raise FileNotFoundError("metric_cache_missing")
                with open(row["source_log"], "rb") as f:
                    frames = pickle.load(f)
                start = row["frame_start"]
                raw = frames[start:start + 14]
                if len(raw) != 14 or raw[3]["token"] != token or raw[3]["log_name"] != row["log_name"]:
                    raise ValueError("manifest/raw log mismatch")
                scene = Scene.from_scene_dict_list(raw, None, 4, 10, SensorConfig.build_no_sensors())
                with lzma.open(row["metric_cache"], "rb") as f:
                    cache = pickle.load(f)
                evaluator = OfficialEvaluator(scene, cache, config)
                meta = {"source": row, "environment": env,
                        "log_sha256": fingerprint(row["source_log"]),
                        "cache_sha256": fingerprint(row["metric_cache"]),
                        "map_root": os.environ.get("NUPLAN_MAPS_ROOT"),
                        "vehicle": vars(evaluator.vehicle)}
            meta["setup_seconds"] = time.perf_counter() - setup_start
            summary = run_scene(evaluator, token, args.output / folder, config, meta, demo)
            results.append({"token": token, "folder": folder, "status": summary["status"],
                "search_status": summary["search_status"],
                "search_started": summary['search_started'],
                "search_unique_candidates": summary['search_unique_candidates'],
                "checked_nominal_pass_count": summary['checked_nominal_pass_count'],
                "gt_score": evaluator.gt.score, "nominal_improvements": len(summary["nominal_improvement_archive"]),
                "diverse_representatives": len(summary["engineering_diverse_representatives"]),
                "seconds": summary["seconds"]})
        except Exception as exc:
            error = {"token": token, "folder": folder, "status": "failed",
                     "search_status": "initialization_or_runtime_error",
                     "error": str(exc), "traceback": traceback.format_exc(),
                     "diagnostics": getattr(exc, 'diagnostics', None)}
            results.append(error)
            (args.output / folder).mkdir(exist_ok=True)
            dump(args.output / folder / "error.json", error)
            print(f"FAILED {token}: {exc}", flush=True)
        dump(args.output / "summary.json", {"requested": len(rows), "processed": len(results),
             "failed": sum(r["status"] == "failed" for r in results), "scenes": results,
             "search_status_counts": dict(Counter(r.get('search_status', 'failed') for r in results)),
             "export_certified": False, "training_export_enabled": False})
    links = ''.join('<tr><td>'+html.escape(r['token'])+'</td><td>'+html.escape(r.get('search_status', r['status']))+'</td><td>'+(
        '<a href="'+r['folder']+'/report.html">轨迹检查</a>' if r['status']!='failed' else
        '<a href="'+r['folder']+'/error.json">失败详情</a>')+'</td></tr>' for r in results)
    (args.output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Offline search</title>'
        '<h1>DrivoR 离线搜索工程检查</h1><p>全部输出仅用于工程检查，未开放训练标签导出。</p><table>'+links+'</table>', encoding="utf-8")
    print(f"Report: {args.output / 'index.html'}", flush=True)
    return 2 if any(r["status"] == "failed" for r in results) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest", help="Build a log-balanced 1..16 scene train manifest")
    p.add_argument("--scene-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--train-tokens", type=Path, required=True)
    p.add_argument("--exclude-tokens", type=Path, action="append", default=[])
    p.add_argument("--count", type=int, choices=range(1, 17), default=16)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--output", type=Path, required=True)
    for name in ["demo", "run"]:
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
        p.add_argument("--seed", type=int, default=20260914)
        p.add_argument("--population", type=int, default=8 if name == "demo" else 32)
        p.add_argument("--generations", type=int, default=2 if name == "demo" else 5)
        if name == "run":
            p.add_argument("--manifest", type=Path, required=True)
            p.add_argument("--map-root", type=Path)
    args = parser.parse_args()
    if args.command == "manifest":
        prepare(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
