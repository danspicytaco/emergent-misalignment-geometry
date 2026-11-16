"""Synthetic replay plumbing only. Real-data parity is a separate explicit CLI run."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduction/replay_anchors.py"
spec = importlib.util.spec_from_file_location("replay_anchors", SCRIPT)
assert spec is not None and spec.loader is not None
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def scratch(self):
        # Never place probes under the system temp dir.
        scratch = Path(os.environ.get("TMPDIR", Path.home() / ".hermes/cache/scratch"))
        scratch.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="em-replay-test-", dir=scratch)
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    def test_help_is_offline_and_does_not_import_models(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                                capture_output=True, text=True, cwd=self.scratch())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--data-root", result.stdout)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_missing_data_is_error_not_skip_or_output(self):
        scratch = self.scratch()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            replay.main(["--anchor", "all", "--data-root", str(scratch / "missing"),
                         "--out-dir", str(scratch / "out")])
        self.assertEqual(error.exception.code, 1)
        self.assertFalse((scratch / "out").exists())

    def test_read_only_root_output_guard(self):
        scratch = self.scratch()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            replay.main(["--data-root", str(scratch), "--out-dir", str(scratch / "out")])
        self.assertFalse((scratch / "out").exists())

    def test_csv_recomputed_and_expected_anchor_only_asserts(self):
        scratch = self.scratch()
        path = scratch / "synthetic.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["idx", "score", "d_layer1"])
            writer.writerows([[0, 3, 1], [1, 2, 2], [2, 1, 3]])
        case = {"id": "synthetic-only", "path": path.name, "layer": 1, "n": 3, "spearman": -1.}
        result = replay.replay_csv(scratch, case, 1e-6)
        self.assertEqual(result["spearman"], -1.)
        with self.assertRaises(ValueError):
            replay.replay_csv(scratch, dict(case, spearman=.5), 1e-6)
        with self.assertRaises(ValueError):
            replay.replay_csv(scratch, dict(case, n=4), 1e-6)

    def test_repeat_recomputed_and_rejects_bad_judgments_or_alignment(self):
        # Eight invented pairs: deliberately not a fixture for any reported result.
        import numpy as np
        scratch = self.scratch()
        d = np.column_stack([np.zeros(8), np.arange(8), np.arange(8)[::-1],
                             np.arange(8)]).astype(np.float32)
        np.save(scratch / "synthetic.npy", d)
        reference = [{"idx": i, "prompt": f"synthetic-p{i}", "response": f"synthetic-r{i}"}
                     for i in range(8)]
        (scratch / "reference.json").write_text(json.dumps(reference))
        records = [dict(row, score=8 - row["idx"], error=False) for row in reference]
        spec = {"distances": "synthetic.npy", "reference": "reference.json",
                "shape": [8, 4], "dtype": "float32", "layers": [1, 2],
                "bin_size": 2, "sort": "default", "drop": 0}
        case = {"id": "synthetic-only", "path": "pass.json", "layer": 1, "mean_rp": -1.}
        (scratch / "pass.json").write_text(json.dumps(records[::-1]))
        result = replay.replay_repeat(scratch, spec, case, 1e-6)
        self.assertAlmostEqual(result["best"]["mean_rp"], -1.)
        self.assertEqual(result["best"]["layer"], 1)
        for change in ({"score": None}, {"score": True}, {"error": True},
                       {"response": "different"}, {"idx": 9}):
            bad = [dict(r) for r in records]
            bad[0].update(change)
            (scratch / "pass.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                replay.replay_repeat(scratch, spec, case, 1e-6)

    def test_config_contract_and_all_seven_anchors(self):
        config = json.loads((ROOT / "configs/reproduction/anchors.json").read_text())
        ids = [c["id"] for c in config["csv"]] + [c["id"] for c in config["repeats"]["passes"]]
        self.assertEqual(len(ids), 7)
        self.assertEqual(len(set(ids)), 7)
        self.assertEqual(config["repeats"]["sort"], "default")
        self.assertEqual(config["repeats"]["layers"], [1, 47])
        self.assertEqual(config["repeats"]["dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
