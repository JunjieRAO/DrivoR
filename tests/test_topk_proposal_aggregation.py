import unittest

import torch

from navsim.planning.training.proposal_selection import select_topk_proposal_indices


class TestTopKProposalAggregation(unittest.TestCase):
    def test_selects_highest_score_from_nc_dac_union(self) -> None:
        no_collision_scores = torch.tensor([[0.9, 0.8, 0.1, 0.0, 0.0]])
        drivable_area_scores = torch.tensor([[0.0, 0.1, 0.9, 0.8, 0.0]])
        pdm_scores = torch.tensor([[0.4, 0.7, 0.6, 0.5, 1.0]])

        selected = select_topk_proposal_indices(
            no_collision_scores,
            drivable_area_scores,
            pdm_scores,
            [1, 2, 5],
        )

        self.assertEqual(selected[1].tolist(), [2])
        self.assertEqual(selected[2].tolist(), [1])
        self.assertEqual(selected[5].tolist(), [4])

    def test_uses_smaller_index_to_break_all_ties(self) -> None:
        no_collision_scores = torch.ones((1, 6))
        drivable_area_scores = torch.ones((1, 6))
        pdm_scores = torch.tensor([[0.1, 0.9, 0.9, 1.0, 1.0, 1.0]])

        selected = select_topk_proposal_indices(
            no_collision_scores,
            drivable_area_scores,
            pdm_scores,
            [2, 4],
        )

        self.assertEqual(selected[2].tolist(), [1])
        self.assertEqual(selected[4].tolist(), [3])

    def test_supports_configured_topks_and_caps_k_at_proposal_count(self) -> None:
        no_collision_scores = torch.arange(10, dtype=torch.float32).repeat(2, 1)
        drivable_area_scores = torch.arange(9, -1, -1, dtype=torch.float32).repeat(2, 1)
        pdm_scores = torch.arange(10, dtype=torch.float32).repeat(2, 1)

        selected = select_topk_proposal_indices(
            no_collision_scores,
            drivable_area_scores,
            pdm_scores,
            [4, 8, 16, 32],
        )

        self.assertEqual(tuple(selected), (4, 8, 16, 32))
        for selected_indices in selected.values():
            self.assertEqual(selected_indices.tolist(), [9, 9])

    def test_rejects_invalid_shapes(self) -> None:
        invalid_shapes = [
            ((2, 3), (2, 3), (3,)),
            ((2, 2), (2, 3), (2, 3)),
            ((2, 3), (1, 3), (2, 3)),
        ]
        for no_collision_shape, drivable_area_shape, pdm_shape in invalid_shapes:
            with self.subTest(
                no_collision_shape=no_collision_shape,
                drivable_area_shape=drivable_area_shape,
                pdm_shape=pdm_shape,
            ):
                with self.assertRaises(ValueError):
                    select_topk_proposal_indices(
                        torch.zeros(no_collision_shape),
                        torch.zeros(drivable_area_shape),
                        torch.zeros(pdm_shape),
                        [4],
                    )

    def test_rejects_non_positive_topk(self) -> None:
        scores = torch.zeros((1, 4))

        with self.assertRaisesRegex(ValueError, "positive top-k"):
            select_topk_proposal_indices(scores, scores, scores, [0])


if __name__ == "__main__":
    unittest.main()