#!/usr/bin/env python3
"""Validate steady autotuning against the measured exhaustive GEMM sweep."""

import argparse
import csv
import pathlib
import statistics


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, ...]] = set()
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
            row = dict(zip(header, fields, strict=True))
            key_fields = (
                "name",
                "backend",
                "block_m",
                "block_n",
                "block_k",
                "threads",
                "stages",
            )
            key = tuple(row[field] for field in key_fields if field in row)
            vendor_row = row["backend"].startswith(("cublas-", "cudnn-"))
            if not vendor_row:
                if key in seen_keys:
                    raise ValueError(f"{path}: duplicate benchmark key {key}")
                seen_keys.add(key)
            p10 = float(row["p10_ms"])
            p50 = float(row["p50_ms"])
            p90 = float(row["p90_ms"])
            if p10 <= 0.0 or not p10 <= p50 <= p90:
                raise ValueError(
                    f"{path}: invalid timing order for {key}: "
                    f"{p10}, {p50}, {p90}"
                )
            rows.append(row)
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
    output_path = args.results_dir / "autotune_validation.csv"
    failures: list[str] = []
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "math_mode",
                "name",
                "exhaustive_best_p50_ms",
                "autotuned_p50_ms",
                "ratio",
                "pass",
            ]
        )
        modes = (
            (
                "shared-fp32",
                "shared-fp32",
                "gemm_autotuned_trial*.csv",
                True,
            ),
            (
                "tensorcore-tf32",
                "tensorcore-tf32",
                "gemm_autotuned_tf32_trial*.csv",
                False,
            ),
        )
        for math_mode, sweep_backend, trial_pattern, include_block_thread in modes:
            best: dict[str, float] = {}
            for row in read_rows(sweep_path):
                if row["backend"] != sweep_backend:
                    continue
                latency = float(row["p50_ms"])
                best[row["name"]] = min(best.get(row["name"], latency), latency)
            if include_block_thread:
                block_thread = median_by_name(
                    sorted(args.results_dir.glob("gemm_block-thread_trial*.csv")),
                    "block-thread",
                )
                for name, latency in block_thread.items():
                    if name in best:
                        best[name] = min(best[name], latency)
            autotuned = median_by_name(
                sorted(args.results_dir.glob(trial_pattern)), "autotuned"
            )
            if not best:
                failures.append(f"{math_mode}: exhaustive sweep has no results")
            for name in sorted(best):
                if name not in autotuned:
                    failures.append(
                        f"{math_mode}/{name}: missing steady autotuned result"
                    )
                    continue
                ratio = autotuned[name] / best[name]
                passed = ratio <= args.limit
                writer.writerow(
                    [
                        math_mode,
                        name,
                        f"{best[name]:.9f}",
                        f"{autotuned[name]:.9f}",
                        f"{ratio:.6f}",
                        passed,
                    ]
                )
                if not passed:
                    failures.append(
                        f"{math_mode}/{name}: ratio {ratio:.4f} exceeds "
                        f"{args.limit:.4f}"
                    )
    if failures:
        raise SystemExit("autotune validation failed: " + "; ".join(failures))
    print(f"autotune validation passed; report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
