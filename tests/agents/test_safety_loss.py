import unittest

import torch

from navsim.agents.drivoR.layers.losses.safety_loss import safety_losses, interpolate_poses


class SafetyLossTest(unittest.TestCase):
    def targets(self):
        coordinates = torch.linspace(-10, 10, 81)
        return {
            "drivable_sdf": (3 - coordinates.abs())[:, None].expand(81, 81)[None],
            "sdf_bounds": torch.tensor([[-10., -10., 10., 10.]]),
            "vehicle": torch.tensor([[4., 2., 0.]]),
            "obstacle_corners": torch.tensor([[[[[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]]]]),
            "obstacle_valid": torch.ones(1, 1, 1, dtype=torch.bool),
        }

    def test_dac_gradient_and_safe_footprint(self):
        targets = self.targets()
        proposals = torch.tensor([[[[0., 3., 0.]]]], requires_grad=True)
        loss, _ = safety_losses(proposals, targets, compute_collision=False)
        loss.backward()
        self.assertGreater(loss.item(), 0)
        self.assertGreater(proposals.grad[0, 0, 0, 1].item(), 0)
        safe, _ = safety_losses(torch.zeros_like(proposals), targets, compute_collision=False)
        self.assertEqual(safe.item(), 0)

    def test_collision_gradient_and_mask(self):
        targets = self.targets()
        proposals = torch.tensor([[[[2.5, 0., 0.]]]], requires_grad=True)
        _, loss = safety_losses(proposals, targets, compute_dac=False)
        loss.backward()
        self.assertGreater(loss.item(), 0)
        self.assertLess(proposals.grad[0, 0, 0, 0].item(), 0)
        targets["obstacle_valid"].zero_()
        _, masked = safety_losses(proposals, targets, compute_dac=False)
        self.assertEqual(masked.item(), 0)

    def test_outside_raster_has_restoring_gradient(self):
        proposals = torch.tensor([[[[20., 0., 0.]]]], requires_grad=True)
        loss, _ = safety_losses(proposals, self.targets(), compute_collision=False)
        loss.backward()
        self.assertGreater(proposals.grad[0, 0, 0, 0].item(), 0)

    def test_heading_wrap(self):
        proposals = torch.tensor([[[[3., 0., 3.1], [6., 0., -3.1]]]])
        interpolated = interpolate_poses(proposals, 4)
        self.assertGreater(interpolated[0, 0, 2, 2].item(), 3.)

    def test_all_candidates_normalize_loss_and_gradient(self):
        cases = (
            ("dac", [0., 0., 0.], [0., 3., 0.], {"compute_collision": False}, 0),
            ("collision", [6., 0., 0.], [2.5, 0., 0.], {"compute_dac": False}, 1),
        )
        for name, safe_pose, unsafe_pose, options, loss_index in cases:
            with self.subTest(loss=name):
                single = torch.tensor(unsafe_pose).reshape(1, 1, 1, 3).requires_grad_()
                candidates = torch.tensor(safe_pose).reshape(1, 1, 1, 3).repeat(1, 64, 1, 1)
                candidates[:, :1] = single.detach()
                candidates.requires_grad_()
                single_loss = safety_losses(single, self.targets(), **options)[loss_index]
                candidate_loss = safety_losses(candidates, self.targets(), **options)[loss_index]
                self.assertGreater(single_loss.item(), 0)
                torch.testing.assert_close(candidate_loss, single_loss / 64)
                single_loss.backward()
                candidate_loss.backward()
                torch.testing.assert_close(candidates.grad[:, :1], single.grad / 64)
                self.assertEqual(torch.count_nonzero(candidates.grad[:, 1:]).item(), 0)

    def test_candidate_mean_preserves_batch_weighting(self):
        cases = (
            ("dac", [0., 0., 0.], [0., 3., 0.], [0., 4., 0.], {"compute_collision": False}, 0),
            ("collision", [6., 0., 0.], [2.5, 0., 0.], [2., 0., 0.], {"compute_dac": False}, 1),
        )
        for name, safe_pose, first_pose, second_pose, options, loss_index in cases:
            with self.subTest(loss=name):
                first = torch.tensor(first_pose).reshape(1, 1, 1, 3).requires_grad_()
                second = torch.tensor(second_pose).reshape(1, 1, 1, 3).requires_grad_()
                candidates = torch.tensor(safe_pose).reshape(1, 1, 1, 3).repeat(2, 64, 1, 1)
                candidates[0, :1] = first.detach()[0]
                candidates[1, :3] = second.detach()[0]
                candidates.requires_grad_()
                targets = {key: value.repeat(2, *([1] * (value.ndim - 1)))
                           for key, value in self.targets().items()}
                first_loss = safety_losses(first, self.targets(), **options)[loss_index]
                second_loss = safety_losses(second, self.targets(), **options)[loss_index]
                batch_loss = safety_losses(candidates, targets, **options)[loss_index]
                torch.testing.assert_close(batch_loss, (first_loss + 3 * second_loss) / (2 * 64))
                first_loss.backward()
                second_loss.backward()
                batch_loss.backward()
                torch.testing.assert_close(candidates.grad[0, :1], first.grad[0] / (2 * 64))
                torch.testing.assert_close(candidates.grad[1, :3], second.grad[0].expand(3, -1, -1) / (2 * 64))

    def test_all_safe_candidates_return_zero_with_zero_gradient(self):
        candidates = torch.tensor([6., 0., 0.]).reshape(1, 1, 1, 3).repeat(1, 64, 1, 1).requires_grad_()
        dac_loss, collision_loss = safety_losses(candidates, self.targets())
        self.assertEqual(dac_loss.item(), 0)
        self.assertEqual(collision_loss.item(), 0)
        (dac_loss + collision_loss).backward()
        self.assertTrue(torch.isfinite(candidates.grad).all())
        self.assertEqual(torch.count_nonzero(candidates.grad).item(), 0)

    def test_full_loss_preserves_original_and_backpropagates(self):
        from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss
        proposals = torch.tensor([[[[0., 3., 0.]], [[2.5, 0., 0.]]]], requires_grad=True)
        names = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
                 "ego_progress", "driving_direction_compliance", "comfort", "lane_keeping")
        prediction = {
            "proposals": proposals, "proposal_list": [proposals],
            "pred_logit": {name: torch.zeros(1, 2, requires_grad=True) for name in names},
            "pdm_score": torch.zeros(1, 2),
        }
        for name in ("pred_logit2", "pred_agents_states", "pred_area_logit", "agent_states", "bev_semantic_map"):
            prediction[name] = None
        labels = self.targets()
        def score(targets, trajectories, test):
            return (torch.ones(1, 2), torch.ones(1), torch.ones(1, 2, 8), None, None, None, labels)
        targets = {"trajectory": torch.zeros(1, 1, 3)}
        baseline = DrivoRLoss()(targets, prediction, None, score)
        augmented = DrivoRLoss(dac_weight=1.0, collision_weight=0.5)(targets, prediction, None, score)
        torch.testing.assert_close(
            augmented["loss"], baseline["loss"] + augmented["dac_geometry_loss"]
            + 0.5 * augmented["collision_geometry_loss"],
        )
        augmented["loss"].backward()
        self.assertTrue(torch.isfinite(proposals.grad).all())
        self.assertGreater(proposals.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()