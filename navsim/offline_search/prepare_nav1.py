"""Prepare the 16-scene engineering run using the repository's NAV1 splits.

Generate official metric_cache.MetricCache objects, not train_metric_chache objects.
"""
from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path
import pickle
import traceback

import yaml

from .core import stable_seed
from .run import dump, fingerprint


def select_rows(repo, scene_root, count, seed):
    split_file = repo / 'navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml'
    logs_file = repo / 'navsim/planning/script/config/training/default_train_val_test_log_split.yaml'
    split = yaml.safe_load(split_file.read_text())
    log_split = yaml.safe_load(logs_file.read_text())
    allowed_tokens = set(split['tokens'])
    if not all(isinstance(t, str) for t in allowed_tokens):
        raise ValueError('navtrain token configuration must contain strings')
    allowed_logs = (set(split['log_names']) & set(log_split['train_logs'])) - set(log_split['val_logs'])
    groups = {}
    for log in sorted(allowed_logs, key=lambda x: stable_seed(seed, x)):
        path = scene_root / (log + '.pkl')
        if not path.is_file():
            continue
        with path.open('rb') as f:
            frames = pickle.load(f)
        choices = []
        for i in range(3, len(frames) - 10):
            frame = frames[i]
            if frame['log_name'] == log and frame['token'] in allowed_tokens and frame['roadblock_ids']:
                choices.append({'token': frame['token'], 'log_name': frame['log_name'],
                                'source_log': str(path.resolve()), 'frame_start': i - 3,
                                'declared_split': 'train', 'metric_cache': None,
                                'allowlist_sha256': fingerprint(str(split_file)),
                                'log_split_sha256': fingerprint(str(logs_file))})
        if choices:
            groups[log] = sorted(choices, key=lambda r: stable_seed(seed, r['token']))[:2]
        # At most count logs are needed to fill count slots with one scene/log.
        if len(groups) == count:
            break
    rows = []
    for index in range(2):
        for choices in groups.values():
            if index < len(choices) and len(rows) < count:
                rows.append(choices[index])
    if not rows:
        raise ValueError('No matching non-validation navtrain scenes with 5 seconds of future data')
    if len({r['token'] for r in rows}) != len(rows):
        raise ValueError('Duplicate tokens in selected raw logs')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--scene-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, choices=range(1, 17), default=16)
    p.add_argument('--seed', type=int, default=20260914)
    args = p.parse_args()
    # Check imports before creating a run directory; no model/checkpoint required.
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    import os
    map_root = os.environ['NUPLAN_MAPS_ROOT']
    if not Path(map_root).is_dir() or not args.scene_root.is_dir():
        raise FileNotFoundError('map root or trainval logs directory does not exist')
    rows = select_rows(args.repo.resolve(), args.scene_root, args.count, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / 'selected_before_caching.json', rows)
    cache_root = args.output.resolve() / 'official_metric_cache'
    processor = MetricCacheProcessor(str(cache_root), force_feature_computation=False)
    errors = []
    for index, row in enumerate(rows):
        print(f"[cache {index+1}/{len(rows)}] {row['token']}", flush=True)
        try:
            with open(row['source_log'], 'rb') as f:
                frames = pickle.load(f)
            start = row['frame_start']
            scene = Scene.from_scene_dict_list(frames[start:start+14], None, 4, 10, SensorConfig.build_no_sensors())
            scenario = NavSimScenario(scene, map_root=map_root, map_version='nuplan-maps-v1.0')
            processor.compute_metric_cache(scenario)
            path = cache_root / scenario.log_name / scenario.scenario_type / scenario.token / 'metric_cache.pkl'
            with lzma.open(path, 'rb') as f:
                cache = pickle.load(f)
            if not getattr(cache, 'trajectory', None):
                raise ValueError('Cache lacks official PDM reference trajectory')
            if cache.observation._observation_sample_res != 1:
                raise ValueError('Cache is not sampled at 10 Hz')
            row['metric_cache'] = str(path)
        except Exception as exc:
            errors.append({'token': row['token'], 'error': str(exc), 'traceback': traceback.format_exc()})
            print(f"CACHE FAILED: {row['token']}: {exc}", flush=True)
    # Keep all selected scenes, including cache failures, in the denominator.
    for filename, selection in [('engineering_manifest.jsonl', rows), ('one_scene_manifest.jsonl', rows[:1])]:
        with (args.output / filename).open('x', encoding='utf-8') as f:
            for row in selection:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    dump(args.output / 'preparation_summary.json', {'requested': args.count, 'selected': len(rows),
        'validation_logs_excluded': True, 'cache_failures': errors,
        'cache_type': 'official metric_cache.MetricCache', 'seed': args.seed})
    print(f"Prepared {len(rows)} scenes; cache failures: {len(errors)}; output: {args.output}")
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
