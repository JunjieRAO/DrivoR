import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from navsim.visualization.pdm_evaluation import (
    render_failure_visualizations,
    select_failure_visualizations,
)


class TestPdmEvaluationVisualization(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.DataFrame(
            [
                self._row("nc", True, 0.0, 1.0, 0.3),
                self._row("ddc-b", True, 1.0, 0.0, 0.2),
                self._row("ddc-a", True, 1.0, 0.0, 0.2),
                self._row("both", True, 0.0, 0.0, 0.1),
                self._row("passing", True, 1.0, 1.0, 0.0),
                self._row("invalid", False, 0.0, 0.0, 0.0),
            ]
        )

    def test_filters_failures_and_sorts_by_score_then_token(self) -> None:
        selected = select_failure_visualizations(self.results, 3)

        self.assertEqual(selected["token"].tolist(), ["both", "ddc-a", "ddc-b"])
        self.assertEqual(selected["failure_reason"].tolist(), ["nc0_ddc0", "ddc0", "ddc0"])

    def test_respects_zero_and_oversized_limits(self) -> None:
        self.assertTrue(select_failure_visualizations(self.results, 0).empty)
        self.assertEqual(len(select_failure_visualizations(self.results, 100)), 4)

    def test_rejects_negative_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            select_failure_visualizations(self.results, -1)

    @patch("navsim.visualization.pdm_evaluation.add_trajectory_to_bev_ax")
    @patch("navsim.visualization.pdm_evaluation.plot_bev_frame")
    def test_renders_baseline_and_oracle_and_writes_index(
        self,
        plot_bev_frame: MagicMock,
        add_trajectory_to_bev_ax: MagicMock,
    ) -> None:
        axis = MagicMock()
        axis.lines = [MagicMock(), MagicMock()]
        figure = MagicMock()
        figure.savefig.side_effect = lambda path, **_: Path(path).touch()
        plot_bev_frame.return_value = (figure, axis)
        scene = SimpleNamespace(scene_metadata=SimpleNamespace(num_history_frames=4))
        scene_loader = MagicMock()
        scene_loader.get_scene_from_token.return_value = scene
        proposals = [object(), object()]
        predictions = {"both": {"proposals": proposals}}
        selected = select_failure_visualizations(self.results, 1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            index = render_failure_visualizations(
                selected,
                scene_loader,
                predictions,
                output_dir,
            )

            self.assertEqual(add_trajectory_to_bev_ax.call_count, 2)
            self.assertIs(add_trajectory_to_bev_ax.call_args_list[0].args[1], proposals[0])
            self.assertIs(add_trajectory_to_bev_ax.call_args_list[1].args[1], proposals[1])
            self.assertTrue((output_dir / "index.csv").is_file())
            self.assertTrue((output_dir / index.iloc[0]["image"]).is_file())

    @staticmethod
    def _row(token: str, valid: bool, nc: float, ddc: float, score: float) -> dict:
        return {
            "token": token,
            "valid": valid,
            "no_at_fault_collisions": nc,
            "driving_direction_compliance": ddc,
            "score": score,
            "selected_proposal_idx": 0,
            "best_proposal_idx": 1,
            "best_no_at_fault_collisions": 1.0,
            "best_driving_direction_compliance": 1.0,
            "best_score": 0.9,
        }


if __name__ == "__main__":
    unittest.main()