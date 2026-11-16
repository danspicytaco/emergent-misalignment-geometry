"""Catalogue saved summaries without inferring missing scientific measurements."""
import argparse
import json

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from em_geometry.data import data_root, output_root

AN = data_root() / "analysis"
ACT_ORDER = ["last_prompt_tok", "prompt_mean", "response_mean", "response_mean (l2)"]
JUDGE_ALIAS = {"judge_chatbot_verified.json": "kimi-k3 †"}


def judge_tag(path):
    name = Path(path).name
    match = re.match(r"judge_(.+?)_chatbot_", name)
    return JUDGE_ALIAS.get(name, match.group(1) if match else name)


def rows(analysis=None):
    analysis = Path(analysis) if analysis is not None else AN
    out: list[dict] = []
    problems: list[str] = []
    if not analysis.is_dir():
        return out, [f"Missing analysis directory: {analysis}"]
    for directory in sorted(p for p in analysis.iterdir() if p.is_dir()):
        acts, info = {}, None
        summaries = sorted(directory.glob("summary_*.json"))
        if (directory / "summary.json").exists():
            summaries.append(directory / "summary.json")
        for path in summaries:
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                match = re.fullmatch(r"summary_(.+)_(cosine|l2)\.json", path.name)
                if match and isinstance(summary, dict) and "headline" in summary and "inputs" in summary:
                    act, metric = match.groups()
                    value = summary["headline"]["spearman"]
                    inputs = summary["inputs"]
                    centroid = Path(inputs["centroid"]).name
                    if not centroid.endswith(f"_{act}.npy"):
                        problems.append(f"{directory.name}: {act} uses centroid {centroid}")
                    parts = Path(inputs["acts_dir"].replace("\\", "/")).parts
                    if not parts:
                        raise ValueError("empty acts_dir")
                    prompt_set = parts[-2] if len(parts) > 1 else "unreported"
                    info = dict(dir=directory.name, prompts=prompt_set,
                                base=parts[-1].replace("-vllm-hybrid", ""),
                                em=Path(inputs["judge"]).parent.name.replace("-vllm-hybrid", ""),
                                judge=judge_tag(inputs["judge"]), n=summary["n_prompts"],
                                layer=summary.get("headline", {}).get("layer", "unreported"),
                                src="scripts/reproduction/sec3_distance_evilness.py")
                    acts[act if metric == "cosine" else f"{act} (l2)"] = value
                elif isinstance(summary, dict) and isinstance(summary.get("results"), dict) and any(k.startswith("jev_") for k in summary["results"]):
                    results = summary["results"]
                    jev = {k[4:]: v for k, v in results.items() if k.startswith("jev_")}
                    out.append(dict(dir=directory.name, prompts="chatbot_verified", base="unreported",
                                    em="unreported", judge="jev-1.14.0",
                                    n={k: v["n"] for k, v in jev.items()}, layer=summary.get("layer", "unreported"),
                                    acts={k: v["per_prompt_rho"] for k, v in jev.items()},
                                    src="scripts/analysis/judges/jev_outcome_metric.py"))
                elif isinstance(summary, dict) and isinstance(summary.get("baseline"), dict) and "n_numeric" in summary["baseline"] and any(k.startswith("repeat_") for k in summary):
                    # Preserve statistic/layer names; these are not interchangeable with L14 rho.
                    for name, values in summary.items():
                        if name != "baseline" and not re.fullmatch(r"repeat_\d+", name):
                            continue
                        out.append(dict(dir=f"{directory.name}/{name}", prompts="unreported", base="unreported",
                                        em="unreported", judge=", ".join(summary.get("returned_models", [])) or "unreported",
                                        n=values["n_numeric"], layer="see named statistics", acts={},
                                        statistics={k: v for k, v in values.items() if k != "curve"},
                                        src=summary.get("protocol_source", "unreported")))
                else:
                    problems.append(f"Unknown summary format: {path.relative_to(analysis)}")
            except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
                problems.append(f"Unreadable/unsupported summary {path.relative_to(analysis)}: {error}")
        if info is not None:
            info["acts"] = acts
            out.append(info)
    return out, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=AN)
    parser.add_argument("--out", type=Path, default=output_root() / "legacy" / "CATALOG.md")
    args = parser.parse_args()
    records, problems = rows(args.analysis)
    lines = ["# Saved-results catalogue", "", f"Inputs: `{args.analysis}`.",
             "Missing provenance is marked unreported; named repeatability statistics retain their layers.", ""]
    for record in records:
        lines += [f"## {record['dir']}", "", "```json", json.dumps(record, indent=2, ensure_ascii=False), "```", ""]
    lines += ["## Warnings", "", *([f"- {p}" for p in problems] or ["None."])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(records)} rows, {len(problems)} warnings")


if __name__ == "__main__":
    main()
