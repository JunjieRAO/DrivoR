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
        r=terminal_speed_check(state,gt)
        assert r['passed'] and r['delta_mps']==delta
        state[3]+=.001
        assert not terminal_speed_check(state,gt)['passed']


def test_similar_improvements_keep_best():
    from test_engineering import fixture_candidate
    from navsim.offline_search.core import representatives
    import copy
    gt=fixture_candidate(np.zeros(10))
    a=copy.deepcopy(gt); a.id='a'; a.metrics['score']=.98
    b=copy.deepcopy(gt); b.id='b'; b.metrics['score']=.99
    assert [c.id for c in representatives([a,b],gt)]==['b']
    assert representatives([],gt)==[]
