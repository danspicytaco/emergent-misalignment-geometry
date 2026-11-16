"""Offline CLI/import contracts; no tensors, credentials or providers required."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = [
    "scripts/analysis/binning/binned_vs_per_prompt.py",
    "scripts/analysis/binning/binning_sensitivity.py",
    "scripts/analysis/binning/plot_bin_scatter.py",
    "scripts/analysis/binning/plot_layer_sweep.py",
    "scripts/analysis/binning/plot_sizes.py",
    "scripts/analysis/figures/plot_base_control.py",
    "scripts/analysis/figures/plot_evilness_vs_distance.py",
    "scripts/analysis/figures/plot_figure2.py",
    "scripts/analysis/figures/plot_figure3.py",
    "scripts/analysis/figures/plot_lexical_control.py",
    "scripts/analysis/figures/plot_model_shapes.py",
    "scripts/analysis/figures/plot_scaling.py",
    "scripts/analysis/figures/plot_scaling_split.py",
    "scripts/analysis/judges/jev_decomposition.py",
    "scripts/analysis/judges/jev_filter_check.py",
    "scripts/analysis/judges/jev_outcome_judge.py",
    "scripts/analysis/judges/jev_outcome_metric.py",
    "scripts/analysis/judges/judge_selection_and_error.py",
    "scripts/analysis/judges/plot_judge_followups.py",
    "scripts/analysis/stress_tests/betley_checks.py",
    "scripts/analysis/stress_tests/judge_schemes.py",
    "scripts/analysis/stress_tests/judge_schemes_analysis.py",
    "scripts/analysis/stress_tests/plot_judge_schemes.py",
    "scripts/analysis/stress_tests/plot_stress_tests.py",
    "scripts/analysis/stress_tests/topic_control.py",
    "scripts/data/catalog.py",
    "scripts/reproduction/build_chatbot_verified.py",
    "scripts/reproduction/compute_training_centroids_all.py",
    "scripts/reproduction/retry_judge_failed.py",
    "scripts/reproduction/sec3_distance_evilness.py"
]


class LayoutTests(unittest.TestCase):
    def test_canonical_layout_without_legacy_code_copies(self):
        self.assertEqual(len(CANONICAL), 30)
        for relative in CANONICAL:
            ast.parse((ROOT / relative).read_text())
        self.assertEqual(list((ROOT / "scripts").glob("*.py")), [])
        for retired in ("scripts/judges", "scripts/stress_tests", "binning-sensitivity"):
            self.assertEqual(list((ROOT / retired).glob("*.py")), [])
        self.assertFalse((ROOT / "binning-sensitivity/run.sh").exists())

    def test_offline_help_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            # This guard fails immediately if --help attempts GPU/vendor imports or a network call.
            (base / "sitecustomize.py").write_text(
                "import sys, socket\n"
                "class Guard:\n"
                " def find_spec(self, fullname, *args):\n"
                "  if fullname.split('.')[0] in {'torch','transformers','vllm','judge_responses'}:\n"
                "   raise RuntimeError('forbidden help import: '+fullname)\n"
                "sys.meta_path.insert(0, Guard())\n"
                "def blocked(*a, **k): raise RuntimeError('network forbidden in CLI help')\n"
                "socket.create_connection=blocked\nsocket.socket.connect=blocked\n"
            )
            env = {k: v for k, v in os.environ.items() if not any(x in k for x in ('API_KEY', 'TOKEN', 'JEVKEY'))}
            env.update(PYTHONPATH=os.pathsep.join([temp, str(ROOT), env.get("PYTHONPATH", "")]),
                       EM_DATA_DIR=str(base / "missing-inputs"), EM_OUTPUT_DIR=str(base / "outputs"),
                       MPLCONFIGDIR=str(base / "mpl"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            for relative in CANONICAL:
                with self.subTest(script=relative):
                    run = subprocess.run([sys.executable, str(ROOT / relative), "--help"], cwd=temp,
                                         env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertIn("usage:", run.stdout.lower())
            self.assertFalse((base / "outputs").exists())
            self.assertFalse((base / "missing-inputs").exists())

    def test_canonical_helper_import_does_not_run_main(self):
        from scripts.analysis.figures.plot_scaling import best_binned
        self.assertTrue(callable(best_binned))

    def test_catalog_unknown_and_repeatability_formats(self):
        from scripts.data.catalog import rows
        import json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, payload in {
                "unknown": {"unexpected": True},
                "repeat": {"baseline": {"n_numeric": 5, "L24_unbinned_spearman": 0.1},
                           "repeat_1": {"n_numeric": 4, "L24_unbinned_spearman": 0.2}},
                "jev": {"layer": 14, "results": {"jev_response_mean": {"n": 5, "per_prompt_rho": 0.3}}},
            }.items():
                directory = root / name
                directory.mkdir()
                (directory / "summary.json").write_text(json.dumps(payload))
            result, warnings = rows(root)
            self.assertEqual(len(result), 3)
            self.assertEqual(len(warnings), 1)
            self.assertIn("Unknown summary format", warnings[0])
            self.assertEqual(next(r for r in result if r['dir'] == 'repeat/baseline')['acts'], {})

    def test_shell_uses_installed_dependencies(self):
        source = (ROOT / "scripts/analysis/binning/run.sh").read_text()
        self.assertNotIn("uv run", source)
        self.assertNotIn("--with", source)
        self.assertIn("em_geometry.data", source)
        subprocess.run(["bash", "-n", str(ROOT / "scripts/analysis/binning/run.sh")], check=True)

    def test_remote_dataset_filename_stays_remote(self):
        tree = ast.parse((ROOT / "scripts/reproduction/build_chatbot_verified.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "hf_hub_download"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.literal_eval(calls[0].args[1]),
                         "data/train-00000-of-00001-f4a410a7274a9509.parquet")

    def test_saved_study_inputs_cannot_be_output(self):
        with tempfile.TemporaryDirectory() as temp:
            saved = Path(temp) / "saved"
            for name in ("judge_schemes_analysis", "plot_judge_schemes", "topic_control", "judge_schemes"):
                args = ["--inputs", str(saved), "--out", str(saved / "new")]
                if name == "judge_schemes":
                    args += ["--mode", "original"]
                run = subprocess.run([sys.executable, str(ROOT / "scripts/analysis/stress_tests" / f"{name}.py"), *args],
                                     capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn("must be separate", run.stderr)
            self.assertFalse(saved.exists())


if __name__ == "__main__":
    unittest.main()
