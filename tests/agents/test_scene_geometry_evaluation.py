"""No dataset/GPU required: protect checkpoint choice and incomplete-result handling."""
import csv
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("geometry_evaluation", Path(__file__).parents[2] / "scripts/scene_geometry_evaluation.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class EvaluationTests(unittest.TestCase):
    def test_best_selection_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            repo = Path(folder)
            run = repo / "exp/ke/drivor_sg_gen/run"
            config = run / "code/hydra/config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("agent: {}")
            best = run / "lightning_logs/version_0/checkpoints/best-epoch=8.ckpt"
            best.parent.mkdir(parents=True)
            best.touch()
            (best.parent / "last.ckpt").touch()
            self.assertEqual(pipeline.select_checkpoint(repo), (best.resolve(), config.resolve()))
            other = best.with_name("best-epoch=9.ckpt")
            other.touch()
            with self.assertRaises(ValueError):
                pipeline.select_checkpoint(repo)
            with patch.dict(os.environ, {"CHECKPOINT_PATH": str(other)}):
                self.assertEqual(pipeline.select_checkpoint(repo)[0], other.resolve())

    def test_hydra_checkpoint_quote(self):
        self.assertEqual(pipeline.quoted("agent.checkpoint_path", "/a b/best-epoch=8.ckpt"),
                         'agent.checkpoint_path="/a b/best-epoch=8.ckpt"')

    def test_result_coverage_and_validity(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows = [{"token": token, "valid": "True", "score": "0.5"} for token in (
                "scene", "extended_pdm_score_stage_one", "extended_pdm_score_stage_two", "extended_pdm_score_combined")]
            def write():
                with (output / "result.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=("token", "valid", "score"))
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            self.assertEqual(pipeline.verify_results(output, {"scene"})["scenarios"], 1)
            with self.assertRaises(RuntimeError):
                pipeline.verify_results(output, {"scene", "missing"})
            rows[-1]["valid"] = "False"
            write()
            with self.assertRaises(RuntimeError):
                pipeline.verify_results(output, {"scene"})
            rows[-1]["valid"], rows[0]["score"] = "True", "nan"
            write()
            with self.assertRaises(RuntimeError):
                pipeline.verify_results(output, {"scene"})


if __name__ == "__main__":
    unittest.main()
