
import numpy as np
from navsim.offline_search.adapter_data import experiment_speed_limit

def gt():
    s = np.zeros((41,11))
    s[:,0] = np.arange(41)*.5
    s[:,3] = np.linspace(2,6,41)
    return s

def check(x, intersection=True, limit=None):
    return experiment_speed_limit({'limit_mps':limit}, [x,0], intersection, gt())

def test_map_limit_has_priority():
    assert check(5, limit=3)['limit_mps'] == 3

def test_spatial_interpolation():
    sample = check(5.25)
    assert abs(sample['limit_mps'] - 4.05) < 1e-8
    assert sample['source'] == 'spatial_gt_rollout_plus_1'

def test_unknown_non_intersection_is_explicitly_unbounded():
    assert check(25, False)['allowed_unbounded']
    assert check(25, False)['limit_mps'] is None

def test_no_end_extrapolation():
    assert check(21)['reason'] == 'gt_projection_unverifiable'

def test_ambiguous_return_path_rejected():
    s = gt()
    s[21:,0] = np.arange(19,-1,-1)*.5
    assert experiment_speed_limit({'limit_mps':None}, [5,0], True, s)['reason'] == 'gt_projection_unverifiable'

def test_stationary_reference():
    s = gt(); s[:,:4] = 0
    assert experiment_speed_limit({'limit_mps':None}, [0,0], True, s)['limit_mps'] == 1


def test_faster_candidate_uses_observed_gt_tail():
    short = gt()
    assert experiment_speed_limit({'limit_mps':None}, [21,0], True, short)['limit_mps'] is None
    extended = np.zeros((51,11))
    extended[:,0] = np.arange(51)*.5
    extended[:,3] = 5.
    result = experiment_speed_limit({'limit_mps':None}, [21,0], True, extended)
    assert result['limit_mps'] == 6.
    assert experiment_speed_limit({'limit_mps':None}, [26,0], True, extended)['limit_mps'] is None
