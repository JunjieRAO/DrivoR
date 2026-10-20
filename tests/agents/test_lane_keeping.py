import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as functional
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import LineString, box

from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss
from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR.score_module.scorer import Scorer
from navsim.agents.drivoR.score_module.train_pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, WeightedMetricIndex


class LaneKeepingMetricTest(unittest.TestCase):
    def make_scorer(self, deviations, intersections=(), interval=0.1, window=2.0):
        deviations = np.atleast_2d(np.asarray(deviations, dtype=np.float64))
        scorer = PDMScorer(
            TrajectorySampling(num_poses=deviations.shape[1] - 1, interval_length=interval),
            PDMScorerConfig(lane_keeping_horizon_window=window),
        )
        scorer._num_proposals = deviations.shape[0]
        scorer._ego_coords = np.zeros((*deviations.shape, len(BBCoordsIndex), 2))
        scorer._ego_coords[:, :, BBCoordsIndex.CENTER, 0] = np.arange(deviations.shape[1])
        scorer._ego_coords[:, :, BBCoordsIndex.CENTER, 1] = deviations
        scorer._centerline = SimpleNamespace(linestring=LineString([(-1, 0), (100, 0)]))
        scorer._drivable_area_map = PDMDrivableMap(
            [str(index) for index in range(len(intersections))],
            [SemanticMapLayer.INTERSECTION] * len(intersections),
            list(intersections),
        )
        scorer._weighted_metrics = np.zeros((len(WeightedMetricIndex), scorer._num_proposals))
        return scorer

    def scores(self, deviations, **kwargs):
        scorer = self.make_scorer(deviations, **kwargs)
        scorer._calculate_lane_keeping()
        return scorer._weighted_metrics[WeightedMetricIndex.LANE_KEEPING]

    def test_threshold_and_continuous_window(self):
        deviations = np.zeros((5, 41))
        deviations[0] = 0.5
        deviations[1, :19] = 0.6
        deviations[2, :20] = 0.6
        deviations[3, :19] = 0.6
        deviations[3, 20:39] = 0.6
        deviations[4, 21:] = 0.6
        np.testing.assert_array_equal(self.scores(deviations), [1, 1, 0, 1, 0])

    def test_intersections_are_skipped_without_resetting_counter(self):
        deviations = np.full((2, 41), 0.6)
        deviations[1, 21:] = 0.0
        intersection = box(9.5, -2, 20.5, 2)
        np.testing.assert_array_equal(self.scores(deviations, intersections=[intersection]), [0, 1])

    def test_trajectory_inside_intersection_passes(self):
        np.testing.assert_array_equal(
            self.scores(np.ones(41), intersections=[box(-1, -2, 100, 2)]), [1],
        )

    def test_raw_linestring_in_existing_training_cache(self):
        scorer = self.make_scorer([np.zeros(41), np.ones(41)])
        scorer._centerline = scorer._centerline.linestring
        scorer._calculate_lane_keeping()
        np.testing.assert_array_equal(scorer._weighted_metrics[WeightedMetricIndex.LANE_KEEPING], [1, 0])

    def test_non_integral_window_rounds_up(self):
        deviations = np.zeros((2, 11))
        deviations[0, :6] = 0.6
        deviations[1, :7] = 0.6
        np.testing.assert_array_equal(self.scores(deviations, interval=0.3), [1, 0])

    def test_score_proposals_calls_lane_keeping(self):
        scorer = self.make_scorer(np.zeros(41))
        with patch.multiple(
            scorer, _reset=unittest.mock.DEFAULT, _calculate_ego_area=unittest.mock.DEFAULT,
            _calculate_no_at_fault_collision=unittest.mock.DEFAULT,
            _calculate_drivable_area_compliance=unittest.mock.DEFAULT,
            _calculate_driving_direction_compliance=unittest.mock.DEFAULT,
            _calculate_progress=unittest.mock.DEFAULT, _calculate_ttc=unittest.mock.DEFAULT,
            _calculate_is_comfortable=unittest.mock.DEFAULT, _aggregate_scores=unittest.mock.DEFAULT,
        ):
            with patch.object(scorer, "_calculate_lane_keeping", wraps=scorer._calculate_lane_keeping) as calculate:
                scorer.score_proposals(None, None, None, None, None, 1.0)
                calculate.assert_called_once_with()

    def test_lane_keeping_weight_and_legacy_indices(self):
        self.assertEqual(int(WeightedMetricIndex.COMFORTABLE), 2)
        self.assertEqual(int(WeightedMetricIndex.DRIVING_DIRECTION), 3)
        weights = PDMScorerConfig().weighted_metrics_array
        self.assertEqual(weights[WeightedMetricIndex.LANE_KEEPING], 2.0)
        self.assertEqual(weights.sum(), 14.0)


class LaneKeepingHeadTest(unittest.TestCase):
    def make_head(self, double_score=False):
        config = SimpleNamespace(
            b2d=False, proposal_num=3, tf_d_model=8, tf_d_ffn=16,
            double_score=double_score, agent_pred=False, area_pred=False,
            bev_map=False, bev_agent=False, num_poses=8, one_token_per_traj=True,
        )
        return Scorer(config)

    def test_head_and_optional_double_score_shapes(self):
        head = self.make_head(double_score=True)
        logits, secondary, *_ = head(torch.zeros(2, 3, 8, 3), torch.randn(2, 3, 8))
        self.assertEqual(head.score_num, 7)
        self.assertEqual(len(logits), 7)
        self.assertEqual(logits["lane_keeping"].shape, (2, 3))
        self.assertEqual(secondary.shape, (2, 3, 7))

    def test_lane_keeping_loss_backpropagates_without_mutating_targets(self):
        head = self.make_head()
        features = torch.randn(2, 3, 8, requires_grad=True)
        logits, *_ = head(torch.zeros(2, 3, 8, 3), features)
        targets = torch.ones(2, 3, 8)
        targets[:, :, 6] = torch.tensor([[0, 1, 0], [1, 0, 1]])
        original_targets = targets.clone()
        sub_losses, total, *_ = DrivoRLoss().score_loss(
            logits, None, None, None, targets, None, None, None, None,
        )
        self.assertEqual(len(sub_losses), 7)
        torch.testing.assert_close(sub_losses[-1], functional.binary_cross_entropy_with_logits(
            logits["lane_keeping"], targets[:, :, 6],
        ))
        torch.testing.assert_close(total, sum(sub_losses))
        total.backward()
        torch.testing.assert_close(targets, original_targets)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(head.pred_score["lane_keeping"][-1].weight.grad.abs().sum().item(), 0)


class LaneKeepingRankingTest(unittest.TestCase):
    def test_lk_is_an_additive_weighted_metric_and_can_be_disabled(self):
        model = DrivoRModel.__new__(DrivoRModel)
        torch.nn.Module.__init__(model)
        model._config = SimpleNamespace(noc=1, dac=1, ddc=0, ttc=5, ep=5, comfort=2, lk=0)
        logits = {name: torch.zeros(1, 2) for name in (
            "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
            "time_to_collision_within_bound", "ego_progress", "comfort", "lane_keeping",
        )}
        logits["lane_keeping"] = torch.tensor([[-4.0, 4.0]])
        legacy_scores = model._compute_pdm_score(logits)
        torch.testing.assert_close(legacy_scores, torch.full((1, 2), np.log(0.5 * 0.5 * 6.0)))
        self.assertEqual(legacy_scores.argmax(dim=1).item(), 0)
        model._config.lk = 2.0
        scores = model._compute_pdm_score(logits)
        torch.testing.assert_close(scores, (0.5 * 0.5 * (6.0 + 2.0 * logits["lane_keeping"].sigmoid())).log())
        self.assertEqual(scores.argmax(dim=1).item(), 1)
        del model._config.lk
        torch.testing.assert_close(model._compute_pdm_score(logits), legacy_scores)


if __name__ == "__main__":
    unittest.main()