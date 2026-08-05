#!/usr/bin/env python3
"""Aggregate L4 benchmark CSV files and evaluate performance gates."""

import argparse
import csv
import json
import statistics
from pathlib import Path


NUMERIC_FIELDS = (
    "p10_ms",
    "p50_ms",
    "p90_ms",
    "wall_p50_ms",
    "gflops",
    "max_abs",
    "max_rel",
)


def _metadata(path):
    parts = path.stem.split("__")
    if len(parts) != 5 or not parts[4].startswith("trial_"):
        raise ValueError(f"unexpected result filename: {path.name}")
    return {
        "series": parts[0],
        "family": parts[1],
        "case": parts[2],
        "source_mode": parts[3],
        "trial": int(parts[4].removeprefix("trial_")),
    }


def load_records(directory):
    directory = Path(directory)
    records = []
    if not directory.exists():
        return records
    result_paths = (
        path for path in directory.glob("*.csv")
        if path.name.startswith(("timing__", "sweep__"))
    )
    for path in sorted(result_paths):
        metadata = _metadata(path)
        header = None
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if not row:
                    continue
                if row[0] == "kind":
                    header = row
                    continue
                if row[0] != "result" or header is None:
                    continue
                if len(row) != len(header):
                    raise ValueError(f"malformed row in {path}: {row}")
                record = dict(zip(header, row))
                record.update(metadata)
                records.append(record)
    return records


def aggregate(records):
    grouped = {}
    for record in records:
        key = (
            record["series"],
            record["family"],
            record["name"],
            record["backend"],
            record["source_mode"],
        )
        grouped.setdefault(key, []).append(record)

    summaries = []
    for key, rows in sorted(grouped.items()):
        summary = dict(zip(
            ("series", "family", "name", "backend", "source_mode"), key
        ))
        summary["trials"] = len({row["trial"] for row in rows})
        for field in NUMERIC_FIELDS:
            values = [float(row[field]) for row in rows if row.get(field, "")]
            summary[field] = statistics.median(values) if values else None
        p50_values = [float(row["p50_ms"]) for row in rows]
        summary["trial_p50_min_ms"] = min(p50_values)
        summary["trial_p50_max_ms"] = max(p50_values)
        summary["trial_p50_ratio"] = (
            max(p50_values) / min(p50_values) if min(p50_values) > 0 else None
        )
        summaries.append(summary)
    return summaries


def _find(summaries, family, name, backend, source_mode):
    for summary in summaries:
        if (
            summary["series"] == "timing"
            and summary["family"] == family
            and summary["name"] == name
            and summary["backend"] == backend
            and summary["source_mode"] == source_mode
        ):
            return summary
    return None


def _ratio_gate(name, numerator, denominator, relation, threshold):
    if numerator is None or denominator is None:
        return {"name": name, "status": "missing", "actual": None,
                "threshold": threshold, "relation": relation}
    actual = numerator["p50_ms"] / denominator["p50_ms"]
    passed = actual >= threshold if relation == ">=" else actual <= threshold
    return {"name": name, "status": "pass" if passed else "fail",
            "actual": actual, "threshold": threshold, "relation": relation}


def evaluate_gates(summaries):
    gates = []
    gates.append(_ratio_gate(
        "bmm_long custom speedup",
        _find(summaries, "bmm", "bmm_long", "untiled", "untiled"),
        _find(summaries, "bmm", "bmm_long", "block-thread", "block-thread"),
        ">=", 10.0,
    ))
    gates.append(_ratio_gate(
        "conv_resnet_block custom speedup",
        _find(summaries, "conv", "conv_resnet_block", "untiled", "untiled"),
        _find(summaries, "conv", "conv_resnet_block", "block-thread",
              "block-thread"),
        ">=", 10.0,
    ))

    for name in ("bmm_bert", "bmm_long", "bmm_value", "bmm_irregular"):
        gates.append(_ratio_gate(
            f"{name} vendor/direct ratio",
            _find(summaries, "bmm", name, "vendor", "vendor"),
            _find(summaries, "bmm", name, "cublas-pedantic-fp32", "vendor"),
            "<=", 5.0,
        ))
    for name in (
        "conv_small",
        "conv_resnet_stem",
        "conv_resnet_block",
        "conv_pointwise",
        "conv_irregular",
    ):
        gates.append(_ratio_gate(
            f"{name} vendor/direct ratio",
            _find(summaries, "conv", name, "vendor", "vendor"),
            _find(summaries, "conv", name, "cudnn-fp32", "vendor"),
            "<=", 5.0,
        ))

    gemm = _find(summaries, "gemm", "gemm_512", "block-thread",
                 "block-thread")
    if gemm is None:
        gates.append({"name": "PR2 GEMM p50 regression", "status": "missing",
                      "actual": None, "threshold": [0.1206, 0.1474],
                      "relation": "within"})
    else:
        actual = gemm["p50_ms"]
        gates.append({
            "name": "PR2 GEMM p50 regression",
            "status": "pass" if 0.1206 <= actual <= 0.1474 else "fail",
            "actual": actual,
            "threshold": [0.1206, 0.1474],
            "relation": "within",
        })
    return gates


def write_summary_csv(path, summaries):
    fields = (
        "series", "family", "name", "backend", "source_mode", "trials",
        *NUMERIC_FIELDS, "trial_p50_min_ms", "trial_p50_max_ms",
        "trial_p50_ratio",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def write_markdown(path, summaries, gates):
    lines = [
        "# L4 Evaluation Summary",
        "",
        "## Performance Gates",
        "",
        "| Gate | Status | Actual | Requirement |",
        "|---|---|---:|---|",
    ]
    for gate in gates:
        actual = "missing" if gate["actual"] is None else f"{gate['actual']:.6g}"
        threshold = gate["threshold"]
        if isinstance(threshold, list):
            requirement = f"{threshold[0]} to {threshold[1]} ms"
        else:
            requirement = f"{gate['relation']} {threshold:g}"
        lines.append(f"| {gate['name']} | {gate['status']} | {actual} | {requirement} |")

    lines.extend([
        "",
        "## Canonical Timings",
        "",
        "| Family | Name | Backend | Source mode | Trials | p50 ms | p10 ms | p90 ms | GFLOP/s | p50 range |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in summaries:
        if row["series"] != "timing":
            continue
        gflops = "" if row["gflops"] is None else f"{row['gflops']:.3f}"
        lines.append(
            f"| {row['family']} | {row['name']} | {row['backend']} | "
            f"{row['source_mode']} | {row['trials']} | {row['p50_ms']:.6f} | "
            f"{row['p10_ms']:.6f} | {row['p90_ms']:.6f} | {gflops} | "
            f"{row['trial_p50_min_ms']:.6f}-{row['trial_p50_max_ms']:.6f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--require-passing", action="store_true")
    args = parser.parse_args()

    records = load_records(args.results_dir / "raw")
    records.extend(load_records(args.results_dir / "sweeps"))
    summaries = aggregate(records)
    gates = evaluate_gates(summaries)
    overall = "pass" if all(gate["status"] == "pass" for gate in gates) else "fail"

    write_summary_csv(args.results_dir / "summary.csv", summaries)
    payload = {"overall_status": overall, "gates": gates}
    (args.results_dir / "gates.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.results_dir / "evaluation_summary.md", summaries, gates)
    print(f"evaluation_status={overall}")
    if args.require_passing and overall != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
