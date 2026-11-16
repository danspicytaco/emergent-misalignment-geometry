#!/usr/bin/env python3
"""Replay retained numeric evidence, without model imports, network or credentials.

Expected anchors are assertions only. Every observed statistic is recomputed
from CSV distances/scores or the float32 distance array and saved judgments.
This does not reproduce acquisition, activation extraction or tensor distances.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

# Direct source-checkout invocation is independent of the shell working directory.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from em_geometry.metrics import correlations, equal_count_sweep, residual_layers
from em_geometry.records import align_legacy_pairs
import numpy as np


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: observed {actual!r}, expected {expected!r}")


def require_close(actual, expected, tolerance, label):
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"{label}: observed {actual!r}, expected {expected!r} +/- {tolerance}")


def replay_csv(root, case, tolerance):
    with (Path(root) / case["path"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require_equal(len(rows), case["n"], f"{case['id']} count")
    indices = [int(r["idx"]) for r in rows]
    require_equal(len(set(indices)), len(rows), f"{case['id']} unique indices")
    layer = case["layer"]
    c = correlations(np.array([float(r[f"d_layer{layer}"]) for r in rows]),
                     np.array([float(r["score"]) for r in rows]))
    require_close(c.spearman, case["spearman"], tolerance, f"{case['id']} spearman")
    return {"id": case["id"], "input": case["path"], "n": len(rows),
            "layer": layer, "spearman": c.spearman, "pearson": c.pearson}


def replay_repeat(root, spec, case, tolerance):
    root = Path(root)
    distances = np.load(root / spec["distances"], allow_pickle=False)
    require_equal(list(distances.shape), spec["shape"], "distance shape")
    require_equal(str(distances.dtype), spec["dtype"], "distance dtype")
    layers = residual_layers(distances)
    require_equal(layers, list(range(spec["layers"][0], spec["layers"][1] + 1)),
                  "eligible residual layers")
    require_equal(len(distances) % spec["bin_size"], spec["drop"], "dropped remainder")
    reference = read_json(root / spec["reference"])
    records = align_legacy_pairs(reference, read_json(root / case["path"]),
                                 expected_count=len(distances))
    scores = []
    for row in records:
        score = row.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not np.isfinite(score) or row.get("error")):
            raise ValueError(f"invalid numeric judgment at idx={row['idx']}")
        scores.append(float(score))
    curve, best = equal_count_sweep(distances, np.array(scores), spec["bin_size"],
                                    sort=spec["sort"])
    require_equal(best["layer"], case["layer"], f"{case['id']} best layer")
    require_close(best["mean_rp"], case["mean_rp"], tolerance, f"{case['id']} mean_rp")
    return {"id": case["id"], "input": case["path"], "n": len(scores),
            "distance_dtype": str(distances.dtype), "sort": spec["sort"],
            "bin_size": spec["bin_size"], "n_bins": len(scores) // spec["bin_size"],
            "dropped": spec["drop"], "eligible_layers": layers,
            "pair_alignment": "exact idx, prompt and response match against reference",
            "best": best, "curve": curve}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", default="all", help="all or anchor id from config")
    parser.add_argument("--data-root", required=True, type=Path,
                        help="read-only evidence root containing analysis/ and released_14b/")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="output directory outside the read-only evidence root")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reproduction/anchors.json")
    args = parser.parse_args(argv)
    try:
        data_root, out_dir = args.data_root.resolve(), args.out_dir.resolve()
        if out_dir == data_root or data_root in out_dir.parents:
            raise ValueError("out-dir must be outside the read-only data-root")
        config = read_json(args.config)
        known = [c["id"] for c in config["csv"]] + [c["id"] for c in config["repeats"]["passes"]]
        if args.anchor != "all" and args.anchor not in known:
            raise ValueError(f"unknown anchor {args.anchor!r}; choose from {known}")
        results = []
        for case in config["csv"]:
            if args.anchor in ("all", case["id"]):
                results.append(replay_csv(data_root, case, config["statistic_atol"]))
        for case in config["repeats"]["passes"]:
            if args.anchor in ("all", case["id"]):
                results.append(replay_repeat(data_root, config["repeats"], case,
                                             config["statistic_atol"]))
        require_equal(len(results), len(known) if args.anchor == "all" else 1, "anchor count")
        # Write only after every selected input and assertion has passed.
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {"scope": config["scope"], "data_root": str(data_root),
                  "statistic_atol": config["statistic_atol"], "anchors": results}
        (out_dir / "replay_anchors.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for result in results:
            print(json.dumps({k: v for k, v in result.items() if k not in ("curve", "eligible_layers")}))
        print(f"PASS: {len(results)} anchors; report: {out_dir / 'replay_anchors.json'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"replay failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
