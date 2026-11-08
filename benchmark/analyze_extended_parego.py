"""Analyze all problem/batch-size cases of the paired scalarization extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = {
    "per_batch": "One weight vector per batch",
    "per_candidate": "One per candidate",
}
COLORS = {"per_batch": "#0072b2", "per_candidate": "#d55e00"}


def load_case(root, q, problem, seeds, rounds):
    frames = []
    audits = []
    for seed in seeds:
        candidates = {}
        for policy in POLICIES:
            stem = root / f"q{q}" / f"{problem}_{seed}_{policy}"
            frame = pd.read_csv(stem.with_suffix(".csv"))
            assert len(frame) == rounds + 1, stem
            assert frame.iteration.tolist() == list(range(rounds + 1)), stem
            assert (
                np.isfinite(frame[["hypervolume", "ask_seconds", "fit_seconds"]])
                .all()
                .all()
            )
            assert (frame.hypervolume.diff().dropna() >= -1e-10).all(), stem
            frame["q"] = q
            frame["additional_evaluations"] = frame.iteration * q
            frame["cumulative_ask_seconds"] = frame.ask_seconds.cumsum()
            frame["cumulative_compute_seconds"] = (
                frame.ask_seconds + frame.fit_seconds
            ).cumsum()
            frames.append(frame)
            candidates[policy] = pd.read_csv(stem.with_suffix(".candidates.csv"))
            assert len(candidates[policy]) == rounds * q, stem
        for iteration in range(1, rounds + 1):
            weight_groups = {
                policy: records.loc[records.iteration == iteration, "weights"]
                for policy, records in candidates.items()
            }
            assert weight_groups["per_batch"].nunique() == 1
            assert weight_groups["per_candidate"].nunique() == q
            assert (
                weight_groups["per_batch"].iloc[0]
                == weight_groups["per_candidate"].iloc[0]
            )
        input_keys = [key for key in candidates["per_batch"] if key.startswith("x")]
        difference = np.max(
            np.abs(
                candidates["per_batch"].iloc[0][input_keys].astype(float)
                - candidates["per_candidate"].iloc[0][input_keys].astype(float)
            )
        )
        audits.append(
            {
                "problem": problem,
                "q": q,
                "seed": seed,
                "first_candidate_max_difference": float(difference),
            }
        )
    data = pd.concat(frames, ignore_index=True)
    initial = data[data.iteration == 0].pivot(
        index="seed", columns="policy", values="hypervolume"
    )
    assert np.allclose(initial.per_batch, initial.per_candidate, atol=1e-12)
    return data, audits


def plot_cases(data, manifest, output, q, metric, title, ylabel):
    problems = manifest["problems"]
    columns = 3
    rows = (len(problems) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows), squeeze=False)
    seed_count = len(manifest["seeds"])
    indices = np.random.default_rng(581).integers(
        0, seed_count, size=(10000, seed_count)
    )
    for axis, problem in zip(axes.flat, problems):
        subset = data[(data.q == q) & (data.problem == problem)]
        for policy in POLICIES:
            pivot = (
                subset[subset.policy == policy]
                .pivot(index="seed", columns="additional_evaluations", values=metric)
                .sort_index()
            )
            values = pivot.to_numpy()
            low, high = np.quantile(
                values[indices].mean(axis=1), [0.025, 0.975], axis=0
            )
            axis.plot(
                pivot.columns,
                values.mean(axis=0),
                label=POLICIES[policy],
                color=COLORS[policy],
                linewidth=2,
            )
            axis.fill_between(
                pivot.columns, low, high, color=COLORS[policy], alpha=0.15
            )
        axis.set_title(problem)
        axis.set_xlabel("Additional function evaluations")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in list(axes.flat)[len(problems) :]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"{title}: q={q}, {seed_count} paired seeds\nMeans and 95% bootstrap confidence intervals"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    root = options.directory
    manifest = json.loads((root / "manifest.json").read_text())
    frames, audits, comparisons, summaries = [], [], [], []
    seed_count = len(manifest["seeds"])
    bootstrap = np.random.default_rng(4387).integers(
        0, seed_count, size=(20000, seed_count)
    )
    for q in manifest["batch_sizes"]:
        rounds = manifest["additional_evaluations"] // q
        for problem in manifest["problems"]:
            data, audit = load_case(root, q, problem, manifest["seeds"], rounds)
            frames.append(data)
            audits.extend(audit)
            final = data[data.iteration == rounds]
            paired = final.pivot(
                index="seed", columns="policy", values="hypervolume"
            ).sort_index()
            differences = (paired.per_candidate - paired.per_batch).to_numpy()
            low, high = np.quantile(differences[bootstrap].mean(axis=1), [0.025, 0.975])
            shared_mean = paired.per_batch.mean()
            comparisons.append(
                {
                    "problem": problem,
                    "q": q,
                    "seeds": seed_count,
                    "mean_batch_hv": shared_mean,
                    "mean_candidate_hv": paired.per_candidate.mean(),
                    "relative_change_pct": 100 * differences.mean() / shared_mean
                    if shared_mean
                    else np.nan,
                    "paired_hv_difference": differences.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "candidate_wins": int((differences > 1e-10).sum()),
                    "batch_wins": int((differences < -1e-10).sum()),
                    "ties": int((np.abs(differences) <= 1e-10).sum()),
                    "wilcoxon_p": float(wilcoxon(differences).pvalue)
                    if np.any(differences)
                    else 1.0,
                }
            )
            for policy in POLICIES:
                ending = final[final.policy == policy]
                timed = data[(data.policy == policy) & (data.iteration > 0)]
                summaries.append(
                    {
                        "problem": problem,
                        "q": q,
                        "policy": policy,
                        "mean_final_hv": ending.hypervolume.mean(),
                        "sd_final_hv": ending.hypervolume.std(),
                        "mean_hv_gain": ending.hv_improvement.mean(),
                        "mean_ask_seconds": timed.ask_seconds.mean(),
                        "mean_total_ask_seconds": ending.cumulative_ask_seconds.mean(),
                        "mean_total_compute_seconds": ending.cumulative_compute_seconds.mean(),
                        "warning_count": int(timed.warnings.sum()),
                    }
                )
    ranked = sorted(comparisons, key=lambda row: row["wilcoxon_p"])
    previous = 0.0
    for i, row in enumerate(ranked):
        previous = max(previous, min(1.0, row["wilcoxon_p"] * (len(ranked) - i)))
        row["holm_p"] = previous
    all_data = pd.concat(frames, ignore_index=True)
    pd.DataFrame(audits).to_csv(root / "pairing_audit.csv", index=False)
    pd.DataFrame(comparisons).to_csv(root / "paired_comparisons.csv", index=False)
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    all_data.to_csv(root / "all_results.csv", index=False)
    for q in manifest["batch_sizes"]:
        plot_cases(
            all_data,
            manifest,
            root / f"quality_q{q}.png",
            q,
            "hv_improvement",
            "Scalarization policy and hypervolume gain",
            "Hypervolume improvement",
        )
        plot_cases(
            all_data,
            manifest,
            root / f"runtime_q{q}.png",
            q,
            "cumulative_ask_seconds",
            "Scalarization policy and candidate-generation time",
            "Cumulative ask time (s)",
        )
    lines = [
        "# Extended ParEGO scalarization benchmark",
        "",
        f"{seed_count} paired seeds for each problem and batch size; "
        f"{manifest['initial']} initial observations and {manifest['additional_evaluations']} additional evaluations. "
        "Both policies use sequential, pending-aware qLogNEI. Only the frequency of drawing a new scalarization changes.",
        "",
        "## Design",
        "",
        "- "
        + "; ".join(
            f"q={q}: {manifest['additional_evaluations'] // q} rounds"
            for q in manifest["batch_sizes"]
        )
        + f". All end at {manifest['initial'] + manifest['additional_evaluations']} observations.",
        "- 128 MC samples, four restarts, 128 raw samples, maxiter=200; noiseless objectives, all minimized.",
        f"- {manifest['workers']} workers with one CPU thread each; BoTorch {manifest['botorch']}.",
        "- Initial designs and first scalarization per round are matched between policies. "
        "Separate random streams control weights, MC samplers, and optimizer initialization.",
        "- The original q=8 runs on DTLZ2, DTLZ7, and Branin–Currin are reused from the versioned November 4 archive. "
        "All q=4 runs and q=8 runs on the added problems are new. Runtime comparisons across batch sizes "
        "are affected by historical four-worker versus new eight-worker CPU contention.",
        "- This is an exploratory extension of the original study. All problem/batch-size cases are reported. "
        f"Paired tests use a single Holm correction across all {len(comparisons)} cases.",
        "",
        "| Problem | Inputs | Objectives | Hypervolume reference point |",
        "|---|---:|---:|---|",
    ]
    for name, spec in manifest["problem_specs"].items():
        lines.append(
            f"| {name} | {spec['inputs']} | {spec['objectives']} | {spec['reference']} |"
        )
    lines += ["", "## Quality curves", ""]
    for q in manifest["batch_sizes"]:
        lines += [
            f"### q={q}",
            "",
            f"![Quality curves for q={q}](quality_q{q}.png)",
            "",
        ]
    lines += [
        "## Final paired comparison",
        "",
        "Positive differences favor independent weights per candidate. Intervals bootstrap the paired seed differences "
        "with 20,000 resamples. Two-sided Wilcoxon signed-rank p-values are Holm-adjusted across the full study.",
        "",
        "| Problem | q | Final HV: shared / per-candidate | Change | Wins / losses / ties | Paired difference, 95% CI | Adjusted p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['problem']} | {row['q']} | {row['mean_batch_hv']:.4f} / {row['mean_candidate_hv']:.4f} "
            f"| {row['relative_change_pct']:+.1f}% | {row['candidate_wins']} / {row['batch_wins']} / {row['ties']} "
            f"| {row['paired_hv_difference']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['holm_p']:.4g} |"
        )
    lines += [
        "",
        "## Candidate-generation cost",
        "",
        "| Problem | q | Policy | Mean ask/batch | Total ask time |",
        "|---|---:|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['problem']} | {row['q']} | {POLICIES[row['policy']]} "
            f"| {row['mean_ask_seconds']:.2f} s | {row['mean_total_ask_seconds']:.2f} s |"
        )
    lines += [
        "",
        "## Limits and artifacts",
        "",
        "The results concern short-horizon synthetic front learning. They do not establish a universal "
        "advantage for every preference, noise level, reference point, or evaluation budget. "
        "Warning counts and optimizer retries are retained in the recorded wall times.",
        "",
        "- [Configuration and provenance](manifest.json)",
        "- [All per-round measurements](all_results.csv)",
        "- [Paired statistics](paired_comparisons.csv)",
        "- [Runtime summary](summary.csv)",
        "- [Pairing checks](pairing_audit.csv)",
        "- Raw per-run and per-candidate files are in q4/ and q8/.",
        "- [q=4 runtime curves](runtime_q4.png)",
        "- [q=8 runtime curves](runtime_q8.png)",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
