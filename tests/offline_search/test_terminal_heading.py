import numpy as np
from shapely.geometry import LineString
from navsim.offline_search.geometry import terminal_heading_check

def fixture():
    gt=np.zeros((51,11)); gt[:,0]=np.arange(51)*.5; gt[:,2]=np.deg2rad(2)
    return gt, LineString([[-5,0],[40,0]])

def test_tolerance_and_regression():
    gt,route=fixture()
    assert terminal_heading_check(np.array([21,0,np.deg2rad(3)]),gt,route)['passed']
    assert not terminal_heading_check(np.array([21,0,np.deg2rad(3.1)]),gt,route)['passed']

def test_spatial_not_time_matching():
    gt,route=fixture(); gt[:,2]=np.deg2rad(np.arange(51))
    h=terminal_heading_check(np.array([10,0,np.deg2rad(20)]),gt,route)
    assert abs(h['gt_error_deg']-20)<1e-8

def test_unknown_tail_rejected():
    gt,route=fixture()
    assert terminal_heading_check(np.array([26,0,0]),gt,route)['reason']=='terminal_heading_unverifiable'

def test_heading_wrap():
    gt,route=fixture(); gt[:,2]=2*np.pi+np.deg2rad(2)
    assert terminal_heading_check(np.array([10,0,np.deg2rad(-2)]),gt,route)['passed']
