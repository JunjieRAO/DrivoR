import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from navsim.offline_search.geometry import certify_separation, certify_containment, envelope, vertices
from navsim.offline_search.core import (Candidate, Config, BOUNDS, basis, residual, search,
                                       representatives, trajectory_distance, truncated_sample)
from navsim.offline_search.run import read_allowlist


LOCAL = np.array([[1., .5], [-1., .5], [-1., -.5], [1., -.5]])


def test_between_samples_collision_is_rejected():
    # No contact at either endpoint; paths cross in the middle.
    a, b = np.array([-4., 0, 0]), np.array([4., 0, 0])
    c, d = np.array([0., -4, 0]), np.array([0., 4, 0])
    assert Polygon(vertices(a, LOCAL)).disjoint(Polygon(vertices(c, LOCAL)))
    assert Polygon(vertices(b, LOCAL)).disjoint(Polygon(vertices(d, LOCAL)))
    assert not certify_separation(a, b, LOCAL, c, d, LOCAL)[0]


def test_road_hole_with_all_corners_inside_is_rejected():
    road = Polygon([[-5,-5],[5,-5],[5,5],[-5,5]], holes=[[[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]]])
    from shapely.geometry import Point
    assert all(road.contains(Point(*v)) for v in LOCAL)
    assert not certify_containment(np.zeros(3), np.zeros(3), LOCAL, road)


def test_initial_contact_and_margin_are_rejected():
    a = np.zeros(3)
    for x in [2., 2.24, 2.25]:
        b = np.array([x, 0, 0])
        assert not certify_separation(a, a, LOCAL, b, b, LOCAL, margin=.25)[0]
    b = np.array([2.3, 0, 0])
    assert certify_separation(a, a, LOCAL, b, b, LOCAL, margin=.25)[0]


def test_rotating_envelope_contains_intermediate_vertices():
    from navsim.offline_search.geometry import interpolate_pose
    a, b = np.array([0., 0., 2.9]), np.array([2., 1., -2.7])
    swept = envelope(a, b, LOCAL)
    for t in np.linspace(0, 1, 301):
        assert swept.covers(Polygon(vertices(interpolate_pose(a, b, t), LOCAL)))


def test_far_parallel_motion_and_containment_pass():
    a, b = np.array([0., 0., 0.]), np.array([2., 0., 0.])
    assert certify_separation(a, b, LOCAL, a + [0, 5, 0], b + [0, 5, 0], LOCAL)[0]
    assert certify_containment(a, b, LOCAL, box(-5, -3, 6, 3))


def fixture_candidate(theta):
    states = np.zeros((41, 11))
    states[:, 0] = np.arange(41) * .5
    states[:, 3] = 5 + theta[0]
    states[:, 1] = theta[1]
    poses = states[5::5, :3].copy()
    from navsim.offline_search.core import candidate_id
    gate = {"checked_nominal_pass": True, "data_verifiable": True, "reasons": [],
            "violation": [0] * 6, "max_jerk": 0., "clearance_lower_bound": 2., "events": []}
    return Candidate(candidate_id(poses), poses, states.copy(), states, {"score": .5 + .1 * theta[0]}, gate,
                     theta=theta.copy())


def test_search_deterministic_and_bounds_immutable():
    gt = fixture_candidate(np.zeros(10))
    config = Config(population=8, generations=2)
    a = search(config, "001e15", gt, {"B0": np.zeros(10)}, fixture_candidate)
    b = search(config, "001e15", gt, {"B0": np.zeros(10)}, fixture_candidate)
    assert [x.record() for x in a[0]] == [x.record() for x in b[0]]
    assert a[2] == b[2]
    assert all(np.all(abs(c.theta) <= BOUNDS) for c in a[0])
    assert all(not c.export_certified for c in a[0])
    assert all(c.score >= gt.score + config.min_gain for c in a[1])


def test_no_feasible_elites_cannot_enter_archive():
    def reject(theta):
        c = fixture_candidate(theta)
        c.gate.update(checked_nominal_pass=False, reasons=["collision"])
        return c
    all_c, archive, trace = search(Config(population=4, generations=2), "token", fixture_candidate(np.zeros(10)),
                                    {"B0": np.zeros(10)}, reject)
    assert all_c and not archive
    assert all(row["feasible"] == 0 for row in trace if "generation" in row)


def test_representatives_are_original_members_and_separated():
    gt = fixture_candidate(np.zeros(10))
    cs = [fixture_candidate(np.array([.1 * i, .4 * i] + [0] * 8)) for i in range(1, 8)]
    selected = representatives(cs, gt)
    assert selected
    assert all(any(c is original for original in cs) for c in selected)
    for i, c in enumerate(selected):
        assert all(trajectory_distance(c, other) >= 1 for other in [gt] + selected[:i])


def test_upper_bound_skip_and_exact_one_target_search():
    gt = fixture_candidate(np.zeros(10))
    gt.metrics['score'] = 1.
    assert search(Config(population=4, generations=1), 't', gt, {'B0': np.zeros(10)}, fixture_candidate)[2][0]['status'] == 'gt_upper_bound'
    gt.metrics['score'] = .995
    assert search(Config(population=4, generations=1), 't', gt, {'B0': np.zeros(10)}, fixture_candidate)[0]


def test_fixed_initial_control_residual_and_no_clipping():
    assert np.all(residual(BOUNDS)[0] == 0)
    with pytest.raises(ValueError):
        residual(BOUNDS * 1.001)
    points, fallback = truncated_sample(np.random.default_rng(11), BOUNDS * .99, BOUNDS, 100)
    assert np.all(abs(points) <= BOUNDS)


def test_numeric_tokens_rejected(tmp_path):
    path = tmp_path / 'tokens.json'
    path.write_text('[123]')
    with pytest.raises(ValueError):
        read_allowlist(path)
    path.write_text('["00123", "1e10"]')
    assert read_allowlist(path) == {'00123', '1e10'}


def test_synthetic_report_serializable_and_no_teacher_export(tmp_path):
    from navsim.offline_search.run import DemoEvaluator, run_scene
    config = Config(population=4, generations=1)
    evaluator = DemoEvaluator(config)
    summary = run_scene(evaluator, '</script>demo', tmp_path / 'scene', config, {}, True)
    assert summary['export_certified'] is False
    report = (tmp_path / 'scene' / 'report.html').read_text(encoding='utf8')
    assert '\\u003c/script>demo' in report
    assert summary['score_kind'] == 'synthetic_proxy'
    assert not (tmp_path / 'scene' / 'selected_targets.npz').exists()
    arrays = np.load(tmp_path / 'scene' / 'engineering_candidates.npz', allow_pickle=False)
    assert any(k.endswith('_executed') for k in arrays.files)


def test_manifest_balanced_preserves_missing_cache_and_refuses_overwrite(tmp_path):
    import pickle
    from types import SimpleNamespace
    from navsim.offline_search.run import prepare
    logs, caches = tmp_path / 'logs', tmp_path / 'caches'
    logs.mkdir()
    caches.mkdir()
    allowed = []
    for name in ['log_a', 'log_b']:
        frames = [{'token': f'{name}_{i:03d}', 'log_name': name} for i in range(30)]
        (logs / (name + '.pkl')).write_bytes(pickle.dumps(frames))
        allowed.extend(f['token'] for f in frames[3:8])
    tokens = tmp_path / 'train.json'
    tokens.write_text(json.dumps(allowed))
    args = SimpleNamespace(train_tokens=tokens, exclude_tokens=[], scene_root=logs,
                           cache_root=caches, count=16, seed=123, output=tmp_path / 'manifest.jsonl')
    prepare(args)
    rows = [json.loads(x) for x in args.output.read_text().splitlines()]
    assert len(rows) == 4
    assert all(r['metric_cache'] is None for r in rows)
    assert all(sum(r['log_name'] == log for r in rows) == 2 for log in ['log_a', 'log_b'])
    with pytest.raises(FileExistsError):
        prepare(args)


def test_manifest_rejects_train_validation_overlap(tmp_path):
    from types import SimpleNamespace
    from navsim.offline_search.run import prepare
    path = tmp_path / 'tokens.json'
    path.write_text('["001e05"]')
    with pytest.raises(ValueError, match='overlaps'):
        prepare(SimpleNamespace(train_tokens=path, exclude_tokens=[path]))
