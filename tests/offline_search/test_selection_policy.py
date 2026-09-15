import numpy as np
from navsim.offline_search.core import improvement_met, improvement_possible
from navsim.offline_search.geometry import terminal_speed_check

def test_improvement_boundary():
    assert improvement_met(.950001,.95)
    assert not improvement_met(.95,.95)
    assert not improvement_met(.943,.94)
    assert improvement_met(.945,.94)
    assert improvement_possible(.9999)
    assert not improvement_possible(1.)

def test_terminal_speed_caps():
    state=np.zeros(11)
    for gt,delta in [(0,0),(1,.1),(4,.4),(10,.5)]:
        state[3]=gt+delta
        r=terminal_speed_check(state,gt,gt+3)
        assert r['passed'] and r['delta_mps']==delta
        state[3]+=.001
        assert not terminal_speed_check(state,gt,gt+3)['passed']


def test_similar_improvements_keep_best():
    from test_engineering import fixture_candidate
    from navsim.offline_search.core import representatives
    import copy
    gt=fixture_candidate(np.zeros(10))
    a=copy.deepcopy(gt); a.id='a'; a.metrics['score']=.98
    b=copy.deepcopy(gt); b.id='b'; b.metrics['score']=.99
    assert [c.id for c in representatives([a,b],gt)]==['b']
    assert representatives([],gt)==[]


def test_terminal_speed_only_for_clear_gt_deceleration():
    state=np.zeros(11); state[3]=20
    for initial,final in [(6.7611,6.8704),(5,5),(5.5,5),(5.49,5)]:
        r=terminal_speed_check(state,final,initial)
        assert r['passed'] and not r['enabled'] and r['limit_mps'] is None
    r=terminal_speed_check(state,5,6.1)
    assert r['enabled'] and not r['passed']

def test_six_observed_deceleration_scenes_trigger():
    state=np.zeros(11)
    for initial,final in [(7.65,4.70),(2.36,.44),(4.09,.92),(5.07,1.79),(9.28,7.26),(12.11,9.63)]:
        assert terminal_speed_check(state,final,initial)['enabled']


def test_deceleration_strict_thresholds_and_zero_initial():
    state=np.zeros(11)
    # Exact thresholds must not trigger unless the other branch independently holds.
    for initial,final in [(10,9),(20,18),(4,3.2),(0,0),(0,1),(4,3.3)]:
        assert not terminal_speed_check(state,final,initial)['enabled']
    for initial,final in [(10,8.99),(4,3.19),(.1,0),(4,3)]:
        assert terminal_speed_check(state,final,initial)['enabled']
