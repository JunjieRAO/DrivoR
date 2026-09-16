import torch

from navsim.agents.drivoR.proposal_metrics import proposal_safety_statistics, wta_gt_statistics


def _scores() -> torch.Tensor:
    # Columns: NC, DAC, EP, TTC, comfort, DDC, final score.
    return torch.tensor(
        [
            [
                [1.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.9, 2.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.6, 0.5, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.7, 2.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 1.0, 0.1, 2.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.2, 2.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.3, 2.0, 0.0, 0.0, 0.0],
            ],
        ]
    )


def test_proposal_safety_statistics() -> None:
    statistics = proposal_safety_statistics(_scores(), torch.tensor([1, 0, 0]))

    expected = {
        "proposal_nc_zero_ratio": (1.0, 4.0),
        "proposal_dac_zero_ratio": (1.0, 4.0),
        "scene_any_nc_zero_ratio": (1.0, 2.0),
        "scene_any_dac_zero_ratio": (1.0, 2.0),
        "proposal_ttc_lt1_ratio": (2.0, 4.0),
        "proposal_safe_ratio": (1.0, 4.0),
        "scene_no_safe_proposal_ratio": (1.0, 2.0),
        "safe_proposal_ep_mean": (0.8, 1.0),
        "avoidable_unsafe_selection_ratio": (1.0, 1.0),
    }
    for name, counts in expected.items():
        assert torch.allclose(statistics[name], torch.tensor(counts, dtype=torch.float64))


def test_statistics_aggregate_by_counts_not_batch_ratios() -> None:
    first = proposal_safety_statistics(_scores()[:1], torch.tensor([1]))
    second = proposal_safety_statistics(_scores()[1:], torch.tensor([0, 0]))

    aggregate = first["proposal_safe_ratio"] + second["proposal_safe_ratio"]
    assert torch.allclose(aggregate, torch.tensor([1.0, 4.0], dtype=torch.float64))
    assert torch.isclose(aggregate[0] / aggregate[1], torch.tensor(0.25, dtype=torch.float64))


def test_statistics_zero_denominators_are_explicit() -> None:
    statistics = proposal_safety_statistics(_scores()[2:], torch.tensor([0]))

    for counts in statistics.values():
        assert counts[1] == 0


def test_wta_gt_statistics_track_frequency_and_loss_share() -> None:
    target = torch.zeros((2, 1, 3))
    proposals = torch.tensor(
        [
            [[[3.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
        ]
    ).reshape(2, 2, 1, 3)
    proposal_pdms = torch.tensor([[0.9, 0.1], [0.4, 0.2]])
    gt_pdms = torch.tensor([0.8, 0.5])

    statistics = wta_gt_statistics(proposals, target, proposal_pdms, gt_pdms)

    assert torch.allclose(
        statistics["wta_pdms_better_than_gt_ratio"],
        torch.tensor([1.0, 2.0], dtype=torch.float64),
    )
    assert torch.allclose(
        statistics["wta_better_than_gt_min_loss_ratio"],
        torch.tensor([3.0, 4.0], dtype=torch.float64),
    )
