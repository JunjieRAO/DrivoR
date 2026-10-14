import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


class TrainingLauncherTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.repo = Path(__file__).resolve().parents[2]
        data = self.directory / "data"
        cache = self.directory / "cache"
        for path in (data / "maps", data / "navsim_logs/trainval", data / "sensor_blobs/trainval", cache):
            path.mkdir(parents=True)
        self.env = {
            **os.environ,
            "REPO_ROOT": str(self.repo),
            "DATA_ROOT": str(data),
            "NUPLAN_MAPS_ROOT": str(data / "maps"),
            "TRAIN_METRIC_CACHE_PATH": str(cache),
            "NAVSIM_EXP_ROOT": str(self.directory / "exp"),
            "GPU_IDS": "0",
            "DATALOADER_WORKERS": "1",
            "EXPERIMENT_NAME": "launcher_test",
            "PYTHON_BIN": sys.executable,
            "BACKGROUND": "1",
            "DRY_RUN": "1",
        }
        self.env.pop("LOG_FILE", None)

    def preview(self, overrides=None, arguments=()):
        result = subprocess.run(
            ["bash", str(self.repo / "scripts/nav2_train_safety.sh"), *arguments],
            env={**self.env, **(overrides or {})},
            check=True, capture_output=True, text=True,
        )
        return shlex.split(result.stdout.splitlines()[-1])

    def test_default_nohup_preview(self):
        command = self.preview()
        self.assertEqual(command[:2], ["nohup", sys.executable])
        self.assertEqual(command[-6], ">")
        self.assertEqual(command[-4:], ["2>&1", "<", "/dev/null", "&"])
        log = Path(command[-5])
        self.assertEqual(log.parent, self.directory / "exp/logs")
        self.assertTrue(log.name.startswith("launcher_test_"))
        self.assertEqual(log.suffix, ".log")
        self.assertFalse(log.parent.exists())

    def test_custom_log_path_with_spaces(self):
        log = str(self.directory / "exp/logs/custom log.log")
        command = self.preview({"LOG_FILE": log})
        self.assertEqual(command[-5], log)

    def test_foreground_mode(self):
        command = self.preview({"BACKGROUND": "0"})
        self.assertEqual(command[0], sys.executable)
        self.assertNotIn(">", command)

    def test_configuration_query_stays_foreground(self):
        command = self.preview(arguments=("--cfg", "job", "--resolve"))
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[-3:], ["--cfg", "job", "--resolve"])
        self.assertNotIn(">", command)

    def test_readme_nav2_training_defaults(self):
        command = self.preview({key: "" for key in (
            "GPU_IDS", "DATALOADER_WORKERS", "BATCH_SIZE", "MAX_EPOCHS", "BASE_LR",
            "PERCEPTION_LR_SCALE", "DAC_WEIGHT", "COLLISION_WEIGHT", "SEED",
        )})
        expected = {
            "train_test_split": "navtrain",
            "cache_path": "null",
            "use_cache_without_dataset": "false",
            "train_ckpt_path": "null",
            "trainer.params.max_epochs": "10",
            "+trainer.params.devices": "4",
            "trainer.params.strategy": "ddp",
            "trainer.params.gradient_clip_val": "0.0",
            "dataloader.params.batch_size": "16",
            "dataloader.params.num_workers": "16",
            "dataloader.params.prefetch_factor": "1",
            "agent.checkpoint_path": "''",
            "agent.lr_args.name": "AdamW",
            "agent.lr_args.base_lr": "0.0002",
            "agent.num_gpus": "4",
            "agent.perception_lr_scale": "1.0",
            "agent.loss.prev_weight": "0.0",
            "agent.loss.dac_weight": "0.0",
            "agent.loss.collision_weight": "0.0",
            "seed": "2",
        }
        for key, value in expected.items():
            with self.subTest(parameter=key):
                self.assertIn(f"{key}={value}", command)
        self.assertFalse(any("persistent_workers=" in argument for argument in command))


if __name__ == "__main__":
    unittest.main()