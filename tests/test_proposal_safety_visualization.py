from navsim.visualization.proposal_safety import proposal_is_safe, selected_is_unsafe


def _result(nc: float, dac: float, ttc: float):
    return {
        "no_at_fault_collisions": nc,
        "drivable_area_compliance": dac,
        "time_to_collision_within_bound": ttc,
    }


def test_proposal_is_safe_requires_all_three_metrics() -> None:
    assert proposal_is_safe(_result(1.0, 1.0, 1.0))
    assert not proposal_is_safe(_result(0.0, 1.0, 1.0))
    assert not proposal_is_safe(_result(1.0, 0.0, 1.0))
    assert not proposal_is_safe(_result(1.0, 1.0, 0.0))


def test_selected_is_unsafe_uses_selected_proposal() -> None:
    detail = {
        "selected_proposal_idx": 1,
        "results": [_result(1.0, 1.0, 1.0), _result(1.0, 0.0, 1.0)],
    }

    assert selected_is_unsafe(detail)


def test_selected_filter_requires_an_explicit_zero() -> None:
    detail = {
        "selected_proposal_idx": 0,
        "results": [_result(0.5, 1.0, 1.0)],
    }

    assert not selected_is_unsafe(detail)
    assert not proposal_is_safe(detail["results"][0])