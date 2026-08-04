#!/usr/bin/env python3
"""Tests for the reproducible project-performance figures."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/benchmarks/plot_project_performance.py"
README = ROOT / "README.md"
REPORT = ROOT / "docs/performance_visuals.md"
FIGURE_NAMES = (
    "01_optimization_milestones",
    "02_gemm_scaling",
    "03_operator_latency",
    "04_vendor_efficiency",
    "05_mixed_programs",
    "06_parameter_tuning",
    "07_nsys_bottlenecks",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProjectPerformancePlotTest(unittest.TestCase):
    def run_plotter(self, *arguments: str) -> None:
        subprocess.run([sys.executable, str(SCRIPT), *arguments], cwd=ROOT, check=True)

    def test_source_data_validation(self) -> None:
        self.run_plotter("--check-only")

    def test_outputs_are_complete_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            self.run_plotter("--output-dir", str(output_dir))
            first_hashes = {path.name: digest(path) for path in output_dir.iterdir()}

            self.assertEqual(len(first_hashes), 14)
            for name in FIGURE_NAMES:
                png = output_dir / f"{name}.png"
                svg = output_dir / f"{name}.svg"
                self.assertGreater(png.stat().st_size, 20_000)
                self.assertGreater(svg.stat().st_size, 10_000)
                with png.open("rb") as handle:
                    header = handle.read(24)
                self.assertEqual(int.from_bytes(header[16:20], "big"), 1600)
                self.assertEqual(int.from_bytes(header[20:24], "big"), 900)
                ET.parse(svg)

            self.run_plotter("--output-dir", str(output_dir))
            second_hashes = {path.name: digest(path) for path in output_dir.iterdir()}
            self.assertEqual(first_hashes, second_hashes)

    def test_markdown_figure_links_resolve(self) -> None:
        references = (
            (README, ROOT, 3),
            (REPORT, REPORT.parent, 7),
        )
        for markdown_path, base, expected_count in references:
            text = markdown_path.read_text(encoding="utf-8")
            links = re.findall(r"!\[[^]]*\]\(([^)]+performance/[^)]+)\)", text)
            self.assertEqual(len(links), expected_count)
            for link in links:
                self.assertTrue((base / link).is_file(), f"Broken image link in {markdown_path}: {link}")


if __name__ == "__main__":
    unittest.main()
