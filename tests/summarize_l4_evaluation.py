# RUN: %PYTHON %s %project_source_dir/src/benchmarks/summarize_l4_evaluation.py

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path


def load_module(path):
    spec = importlib.util.spec_from_file_location("l4_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rows(path, rows):
    fields = [
        "kind", "name", "backend", "p10_ms", "p50_ms", "p90_ms",
        "wall_p50_ms", "gflops", "max_abs", "max_rel",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "kind": "result",
                "p10_ms": row[2] * 0.9,
                "p90_ms": row[2] * 1.1,
                "wall_p50_ms": row[2] * 1.01,
                "gflops": 100.0,
                "max_abs": 0.0,
                "max_rel": 0.0,
                "name": row[0],
                "backend": row[1],
                "p50_ms": row[2],
            })


summary = load_module(Path(sys.argv[1]))
with tempfile.TemporaryDirectory() as temporary:
    raw = Path(temporary) / "raw"
    raw.mkdir()
    (raw / "gpu_before_trial_1.csv").write_text(
        "timestamp,name,pstate\n2026-08-04T00:00:00Z,NVIDIA L4,P8\n",
        encoding="utf-8",
    )
    for trial in range(1, 4):
        write_rows(
            raw / f"timing__bmm__all__untiled__trial_{trial}.csv",
            [("bmm_long", "untiled", 100.0)],
        )
        bmm_vendor = []
        for name in ("bmm_bert", "bmm_long", "bmm_value", "bmm_irregular"):
            bmm_vendor.extend([
                (name, "vendor", 2.0),
                (name, "cublas-pedantic-fp32", 1.0),
            ])
        write_rows(
            raw / f"timing__bmm__all__vendor__trial_{trial}.csv",
            bmm_vendor,
        )
        write_rows(
            raw / f"timing__bmm__all__block-thread__trial_{trial}.csv",
            [("bmm_long", "block-thread", 5.0)],
        )

        write_rows(
            raw / f"timing__conv__all__untiled__trial_{trial}.csv",
            [("conv_resnet_block", "untiled", 100.0)],
        )
        conv_vendor = []
        for name in (
            "conv_small", "conv_resnet_stem", "conv_resnet_block",
            "conv_pointwise", "conv_irregular",
        ):
            conv_vendor.extend([
                (name, "vendor", 2.0),
                (name, "cudnn-fp32", 1.0),
            ])
        write_rows(
            raw / f"timing__conv__all__vendor__trial_{trial}.csv",
            conv_vendor,
        )
        write_rows(
            raw / f"timing__conv__all__block-thread__trial_{trial}.csv",
            [("conv_resnet_block", "block-thread", 5.0)],
        )
        write_rows(
            raw / f"timing__gemm__gemm_512__block-thread__trial_{trial}.csv",
            [("gemm_512", "block-thread", 0.134)],
        )

    records = summary.load_records(raw)
    aggregates = summary.aggregate(records)
    gates = summary.evaluate_gates(aggregates)
    assert len(records) == 69
    assert gates
    assert all(gate["status"] == "pass" for gate in gates), gates
    bmm_gate = next(gate for gate in gates if gate["name"] == "bmm_long custom speedup")
    assert bmm_gate["actual"] == 20.0
