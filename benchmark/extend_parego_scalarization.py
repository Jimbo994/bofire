"""Extend the paired scalarization study at matched evaluation budgets.

Reuse the original three q=8 datasets, then run q=4 and q=8 on six problems.
All individual run files are checkpointed and can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import botorch
import torch
from benchmark_parego_scalarization import RuntimeParEGO, problem_spec, run_pair

import bofire


PROBLEMS = ["DTLZ2", "DTLZ7", "BraninCurrin", "DTLZ2-5obj", "ZDT3", "VehicleSafety"]


def write_config(path, configuration):
    if path.exists():
        existing = json.loads(path.read_text())
        if {k: v for k, v in existing.items() if k != "source_commit"} != {
            k: v for k, v in configuration.items() if k != "source_commit"
        }:
            raise ValueError(f"Configuration mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configuration, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark/extended_results")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--problems", nargs="+", choices=PROBLEMS, default=PROBLEMS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--additional-evaluations", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-reuse", action="store_true")
    options = parser.parse_args()
    if not hasattr(RuntimeParEGO, "_get_objectives_and_constraints"):
        raise RuntimeError(
            "The imported BoFire lacks the updated ParEGO helper. Run from this "
            "checkout with PYTHONPATH=. or install this checkout in editable mode."
        )
    if any(options.additional_evaluations % q for q in options.batch_sizes):
        raise ValueError("The evaluation budget must be divisible by every batch size.")
    base = {
        "initial": 16,
        "mc_samples": 128,
        "restarts": 4,
        "raw_samples": 128,
        "maxiter": 200,
        "threads": 1,
        "workers": options.workers,
        "seeds": options.seeds,
        "problems": options.problems,
    }
    manifest = {
        **base,
        "batch_sizes": options.batch_sizes,
        "additional_evaluations": options.additional_evaluations,
        "botorch": botorch.__version__,
        "bofire": bofire.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runner_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_parego_scalarization.py").read_bytes()
        ).hexdigest(),
        "reuse_original_q8": not options.no_reuse,
        "note": "Exploratory extension. Historical q8 runs retain their original CPU-sharing timings.",
        "problem_specs": {},
    }
    for name in options.problems:
        problem, _, reference = problem_spec(name)
        manifest["problem_specs"][name] = {
            "inputs": problem.dim,
            "objectives": len(reference),
            "reference": reference.tolist(),
        }
    write_config(options.output / "manifest.json", manifest)
    configurations = {}
    for q in options.batch_sizes:
        configuration = dict(
            **base,
            batch_size=q,
            iterations=options.additional_evaluations // q,
            output=str(options.output / f"q{q}"),
        )
        write_config(options.output / f"q{q}" / "config.json", configuration)
        configurations[q] = argparse.Namespace(
            **{**configuration, "output": Path(configuration["output"])}
        )
    reused = 0
    if (
        not options.no_reuse
        and 8 in options.batch_sizes
        and options.additional_evaluations == 64
    ):
        archive_path = Path(__file__).with_name("results") / "raw_runs.zip"
        with zipfile.ZipFile(archive_path) as archive:
            for name in options.problems:
                for seed in options.seeds:
                    for policy in ["per_batch", "per_candidate"]:
                        stem = f"{name}_{seed}_{policy}"
                        members = [f"{stem}.csv", f"{stem}.candidates.csv"]
                        if not all(member in archive.namelist() for member in members):
                            continue
                        for member in members:
                            destination = options.output / "q8" / member
                            if not destination.exists():
                                destination.write_bytes(archive.read(member))
                        reused += 1
    print(
        f"Reusing {reused} original q=8 runs; remaining cases will be computed.",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        futures = [
            pool.submit(run_pair, configurations[q], name, seed)
            for seed in options.seeds
            for name in options.problems
            for q in options.batch_sizes
        ]
        for future in as_completed(futures):
            future.result()
    print("Extended study complete", flush=True)


if __name__ == "__main__":
    main()
