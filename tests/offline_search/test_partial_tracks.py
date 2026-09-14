from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import Polygon, Point, box, LineString

from navsim.offline_search.adapter_data import cached_actors, frame_clock, DataAlignmentError, speed_limit_at
from navsim.offline_search.core import Config, search
from navsim.offline_search.geometry import Actor, NominalValidator, vertices, missing_interval_region
from navsim.offline_search.run import DemoEvaluator, run_scene

LOCAL = np.array([[1., .5], [-1., .5], [-1., -.5], [1., -.5]])


def partial_actor(x, y=0):
    poses = np.full((41, 3), np.nan)
    poses[10:31] = [x, y, 0]
    return Actor('partial', poses, LOCAL)


def evaluate(actor, states=None, limit=None):
    validator = NominalValidator(Config(), LOCAL, [actor], box(-20,-10,50,10),
        LineString([[0,0],[30,0]]), 3., speed_limit_fn=limit or (lambda _: 10.))
    if states is None:
        states = np.zeros((41,11))
        states[:,0] = np.arange(41)*.5
        states[:,3] = 5
    metrics = dict.fromkeys(['no_at_fault_collisions','drivable_area_compliance',
        'time_to_collision_within_bound','comfort','driving_direction_compliance'], 1.)
    return validator.check(states, metrics)


def test_far_partial_track_passes_only_with_explicit_conditional_scope():
    gate = evaluate(partial_actor(250))
    assert gate['checked_nominal_pass']
    assert gate['lifecycle_uncertain_interval_count'] > 0
    assert not gate['lifecycle_bound_assumption_used']
    assert gate['export_certified'] is False
    assert 'missing_actor_motion_bound_validation' in gate['pending_gates']


def test_possible_missing_actor_near_candidate_remains_unverifiable():
    gate = evaluate(partial_actor(20, 6))
    assert gate['checked_nominal_pass']
    assert 'actor_lifecycle_unverifiable:partial' not in gate['reasons']
    assert gate['lifecycle_uncertain_interval_count'] > 0


def test_partial_actor_known_interval_collision_is_not_dropped():
    gate = evaluate(partial_actor(10))
    assert not gate['checked_nominal_pass']
    assert 'collision_or_margin_uncertified' in gate['reasons']


def test_single_observation_initial_contact_is_not_exempted():
    actor = partial_actor(250)
    actor.poses[:] = np.nan
    actor.poses[0] = [0,0,0]
    gate = evaluate(actor)
    assert not gate['checked_nominal_pass']
    assert any(e['kind']=='observed_partial_actor_contact' for e in gate['events'])


def test_internal_gap_uses_both_anchors_without_interpolating_missing_poses():
    actor = partial_actor(100)
    actor.poses[20] = np.nan
    region = missing_interval_region(actor, 19, 55, .1)
    assert region.covers(Point(100,0))
    assert not region.covers(Point(130,0))
    assert np.isnan(actor.poses[20]).all()


def test_cache_reconstruction_preserves_presence_mask():
    class Map(dict):
        @property
        def tokens(self): return list(self)
    maps = [Map() for _ in range(41)]
    for i in range(10,31):
        maps[i]['partial'] = Polygon(vertices([i, 6, .1*i], LOCAL))
    class Observation:
        _sample_interval = .1
        _observation_sample_res = 1
        red_light_token = 'red_light'
        def __getitem__(self,i): return maps[i]
    actors, errors = cached_actors(Observation())
    assert not errors and len(actors)==1
    assert actors[0].valid.sum()==21
    assert not actors[0].valid[0] and not actors[0].valid[-1]
    for i in range(10,31):
        assert np.allclose(vertices(actors[0].poses[i], actors[0].local), np.asarray(maps[i]['partial'].exterior.coords)[:-1])


def test_clock_accepts_millisecond_jitter_and_records_actual_timestamps():
    stamp = 1_000_000 + np.arange(11)*500_000
    stamp[1:] += 4000
    info = frame_clock(stamp)
    assert info['old_2ms_check_would_fail']
    assert info['max_absolute_jitter_s'] == pytest.approx(.004)
    assert info['nominal_relative_s'][-1] == 5
    assert info['actual_relative_s'][-1] == pytest.approx(5.004)


@pytest.mark.parametrize('kind', ['reverse', 'missing'])
def test_clock_rejects_wrong_frame_order_and_large_gaps_with_diagnostics(kind):
    stamp = 1_000_000 + np.arange(11)*500_000
    if kind == 'reverse': stamp[4] = stamp[3]
    else: stamp[4:] += 500_000
    with pytest.raises(DataAlignmentError) as exc:
        frame_clock(stamp)
    assert 'intervals_s' in exc.value.diagnostics


def test_speed_missing_does_not_erase_known_speeding():
    calls = iter([10.] * 20 + [None] * 21)
    states = np.zeros((41,11))
    states[:,3] = 12.
    gate = evaluate(partial_actor(250), states=states, limit=lambda _: next(calls))
    assert 'speed_limit_unverifiable' in gate['reasons']
    assert 'speed_limit' in gate['reasons']
    assert len(gate['speed_limit_diagnostics']['exceeded_indices']) == 20
    assert len(gate['speed_limit_diagnostics']['unknown_indices']) == 21


def test_map_limit_diagnostics_keep_missing_connectors():
    lanes = [{'id':'connector', 'layer':'LANE_CONNECTOR', 'on_route':True,
              'limit_mps':None,'polygon':box(-1,-1,1,1)}]
    value = speed_limit_at(lanes, Point(0,0))
    assert value['limit_mps'] is None
    assert value['matches'][0]['id']=='connector'
    assert value['reason']=='missing_map_limit'


def test_search_runs_with_partial_tracks_and_report_has_null_not_nan(tmp_path):
    config = Config(population=4, generations=1)
    evaluator = DemoEvaluator(config)
    evaluator.actors.append(partial_actor(250))
    evaluator.validator = NominalValidator(config, evaluator.ego_local, evaluator.actors,
        evaluator.road, LineString(evaluator.raw_path[:,:2]), 3., speed_limit_fn=lambda _:13.9)
    evaluator.gt = evaluator.evaluate_theta(np.zeros(10))
    evaluator.roundtrip = evaluator.evaluate_theta(np.r_[np.full(5,.005),np.zeros(5)])
    summary = run_scene(evaluator, 'partial_scene', tmp_path/'out', config, {}, True)
    assert summary['search_started']
    assert summary['search_unique_candidates'] > 0
    assert summary['export_certified'] is False
    report = (tmp_path/'out/report.html').read_text(encoding='utf8')
    assert 'NaN' not in report
    arrays = np.load(tmp_path/'out/actor_replay.npz', allow_pickle=False)
    assert arrays['1_observed'].sum() == 21


def test_blocked_status_distinct_from_completed_search(tmp_path):
    evaluator = DemoEvaluator(Config(population=4,generations=1))
    evaluator.unknown = ['non_box_actor_unverifiable:broken']
    summary = run_scene(evaluator, 'broken', tmp_path/'out', evaluator.config, {}, True)
    assert summary['search_status']=='precheck_blocked'
    assert not summary['search_started']
