import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlhf.experiment import (
    compare_parameters,
    config_sha256,
    finalize_experiment,
    initialize_experiment,
    load_parameters,
)


class ExperimentTest(unittest.TestCase):
    def test_hash_is_order_independent(self):
        self.assertEqual(
            config_sha256({"a": 1, "b": 2}), config_sha256({"b": 2, "a": 1})
        )

    def test_compare_reports_only_changed_values(self):
        rows = compare_parameters(
            {"ppo": {"learning_rate": 1e-6, "epochs": 1}},
            {"ppo": {"learning_rate": 3e-7, "epochs": 1}, "reward": {"whiten": True}},
        )
        self.assertEqual(
            [row["parameter"] for row in rows],
            ["ppo.learning_rate", "reward.whiten"],
        )

    def test_manifest_lifecycle_and_directory_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config = {
                "experiment": {"id": "unit-test", "tags": ["test"]},
                "train": {"seed": 839},
            }
            with patch(
                "rlhf.experiment._collect_run_metadata",
                return_value={"run_type": "test"},
            ):
                started = initialize_experiment(
                    output_dir,
                    config,
                    run_type="test",
                    config_path="config.yaml",
                )
            self.assertEqual(started["status"], "running")
            self.assertEqual(load_parameters(output_dir), config)

            finished = finalize_experiment(
                output_dir,
                summary={"score": 1.0},
            )
            self.assertEqual(finished["status"], "completed")
            self.assertEqual(finished["summary"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
