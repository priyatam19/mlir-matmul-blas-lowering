#!/usr/bin/env python3
"""Validate steady autotuning against the measured exhaustive GEMM sweep."""

import argparse
import csv
import pathlib
import statistics


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = next(csv.reader([raw_line]))
        if fields[:2] == ["kind", "name"]:
            header = fields
        elif fields and fields[0] == "result" and header:
            if len(fields) != len(header):
                raise ValueError(
                    f"{path}: row has {len(fields)} fields, expected {len(header)}"
                )
            rows.append(dict(zip(header, fields, strict=True)))
    return rows


def median_by_name(paths: list[pathlib.Path], backend: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for path in paths:
        for row in read_rows(path):
            if row["backend"] == backend:
                values.setdefault(row["name"], []).append(float(row["p50_ms"]))
    return {name: statistics.median(samples) for name, samples in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("--limit", type=float, default=1.05)
    args = parser.parse_args()
    sweep_path = args.results_dir / "profile_sweep.csv"
    sweep = [
        row
        for row in read_rows(sweep_path)
        if row["backend"] == "shared-fp32"
    ]
    best: dict[str, float] = {}
    for row in sweep:
        latency = float(row["p50_ms"])
        best[row["name"]] = min(best.get(row["name"], latency), latency)

    block_thread = median_by_name(
        sorted(args.results_dir.glob("gemm_block-thread_trial*.csv")),
        "block-thread",
    )
    for name, latency in block_thread.items():
        if name in best:
            best[name] = min(best[name], latency)
    autotuned = median_by_name(
        sorted(args.results_dir.glob("gemm_autotuned_trial*.csv")), "autotuned"
    )

    output_path = args.results_dir / "autotune_validation.csv"
    failures: list[str] = []
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["name", "exhaustive_best_p50_ms", "autotuned_p50_ms", "ratio", "pass"]
        )
        for name in sorted(best):
            if name not in autotuned:
                failures.append(f"{name}: missing steady autotuned result")
                continue
            ratio = autotuned[name] / best[name]
            passed = ratio <= args.limit
            writer.writerow(
                [name, f"{best[name]:.9f}", f"{autotuned[name]:.9f}", f"{ratio:.6f}", passed]
            )
            if not passed:
                failures.append(f"{name}: ratio {ratio:.4f} exceeds {args.limit:.4f}")
    if failures:
        raise SystemExit("autotune validation failed: " + "; ".join(failures))
    print(f"autotune validation passed; report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
