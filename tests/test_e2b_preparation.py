import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_target_conditioned_e2b_preparation as runner


class PreparationRunnerTests(unittest.TestCase):
    def run_preparation(self, directory):
        with patch.object(sys, "argv", ["prepare", "--work-root", directory]):
            runner.main()

    def test_order_and_ready_status(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"SOURCE_GIT_REVISION": "a" * 40}), \
                patch.object(runner.subprocess, "run") as command:
            self.run_preparation(directory)
            status = json.loads((Path(directory) / "preparation_status.json").read_text())
            self.assertEqual(status["status"], "ready")
            self.assertEqual(status["completed_steps"], ["check", "restore", "features", "bundles", "verify"])
            self.assertEqual(command.call_count, 5)
            self.assertFalse(status["real_outcome_experiments_started"])

    def test_failed_step_stops_pipeline(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"SOURCE_GIT_REVISION": "a" * 40}), \
                patch.object(runner.subprocess, "run", side_effect=[None, subprocess.CalledProcessError(1, "restore")]) as command:
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_preparation(directory)
            status = json.loads((Path(directory) / "preparation_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["step"], "restore")
            self.assertEqual(status["completed_steps"], ["check"])
            self.assertEqual(command.call_count, 2)

    def test_missing_revision_does_not_start_commands(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"SOURCE_GIT_REVISION": ""}), \
                patch.object(runner.subprocess, "run") as command:
            with self.assertRaises(RuntimeError):
                self.run_preparation(directory)
            status = json.loads((Path(directory) / "preparation_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
