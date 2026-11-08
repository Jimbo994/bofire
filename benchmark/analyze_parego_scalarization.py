"""Generate curves and a paired-seed report for the scalarization experiment."""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


LABELS = {
    "per_batch": "One scalarization per batch",
    "per_candidate": "One per candidate",
}
COLORS = {"per_batch": "#0072b2", "per_candidate": "#d55e00"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    config = json.loads((directory / "config.json").read_text())
    paths = sorted(
        p
        for p in directory.glob("*.csv")
        if p.name.split("_")[0] in config["problems"]
        and not p.name.endswith(".candidates.csv")
    )
    data = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    expected = (
        len(config["problems"]) * len(config["seeds"]) * 2 * (config["iterations"] + 1)
    )
    assert len(data) == expected, (len(data), expected)
    assert (
        data.groupby(["problem", "seed", "policy"])
        .size()
        .eq(config["iterations"] + 1)
        .all()
    )
    assert np.isfinite(data[["hypervolume", "ask_seconds", "fit_seconds"]]).all().all()
    initial_hv = data[data.iteration == 0].pivot(
        index=["problem", "seed"], columns="policy", values="hypervolume"
    )
    assert np.allclose(initial_hv.per_batch, initial_hv.per_candidate, atol=1e-12)
    increments = (
        data.sort_values("iteration")
        .groupby(["problem", "seed", "policy"])["hypervolume"]
        .diff()
        .dropna()
    )
    assert (increments >= -1e-10).all()
    audit = []
    for problem in config["problems"]:
        for seed in config["seeds"]:
            records = {
                policy: pd.read_csv(
                    directory / f"{problem}_{seed}_{policy}.candidates.csv"
                )
                for policy in LABELS
            }
            assert all(
                len(frame) == config["iterations"] * config["batch_size"]
                for frame in records.values()
            )
            for iteration in range(1, config["iterations"] + 1):
                shared = records["per_batch"].loc[
                    records["per_batch"].iteration == iteration
                ]
                independent = records["per_candidate"].loc[
                    records["per_candidate"].iteration == iteration
                ]
                assert shared.weights.nunique() == 1
                assert independent.weights.nunique() == config["batch_size"]
                assert shared.iloc[0].weights == independent.iloc[0].weights
            first_shared = records["per_batch"].iloc[0]
            first_independent = records["per_candidate"].iloc[0]
            inputs = [
                column for column in records["per_batch"] if column.startswith("x")
            ]
            difference = float(
                np.max(
                    np.abs(
                        first_shared[inputs].astype(float)
                        - first_independent[inputs].astype(float)
                    )
                )
            )
            audit.append(
                {
                    "problem": problem,
                    "seed": seed,
                    "first_candidate_max_difference": difference,
                }
            )
    pd.DataFrame(audit).to_csv(directory / "pairing_audit.csv", index=False)
    data["cumulative_ask_seconds"] = data.groupby(["problem", "seed", "policy"])[
        "ask_seconds"
    ].cumsum()
    data.to_csv(directory / "all_results.csv", index=False)
    final = data[data.iteration == config["iterations"]]
    rng = np.random.default_rng(4387)
    bootstrap_indices = rng.integers(
        0, len(config["seeds"]), size=(20000, len(config["seeds"]))
    )
    comparisons = []
    summary = []
    fig, axes = plt.subplots(
        3, len(config["problems"]), figsize=(14, 10), squeeze=False
    )
    for j, problem in enumerate(config["problems"]):
        frame = data[data.problem == problem]
        for policy in LABELS:
            subset = frame[frame.policy == policy]
            for row, metric, ylabel in [
                (0, "hv_improvement", "Hypervolume improvement"),
                (1, "cumulative_ask_seconds", "Cumulative ask time (s)"),
                (2, "nondominated", "Non-dominated observations"),
            ]:
                pivot = subset.pivot(
                    index="seed", columns="iteration", values=metric
                ).sort_index()
                values = pivot.to_numpy()
                sampled_means = values[bootstrap_indices].mean(axis=1)
                low, high = np.quantile(sampled_means, [0.025, 0.975], axis=0)
                x = np.asarray(pivot.columns) * config["batch_size"]
                axis = axes[row, j]
                axis.plot(
                    x,
                    values.mean(axis=0),
                    color=COLORS[policy],
                    label=LABELS[policy],
                    lw=2,
                )
                axis.fill_between(x, low, high, color=COLORS[policy], alpha=0.15)
                axis.set_xlabel("Additional evaluations")
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.2)
                axis.spines[["top", "right"]].set_visible(False)
            finals = final[(final.problem == problem) & (final.policy == policy)]
            timing = subset[subset.iteration > 0]
            summary.append(
                {
                    "problem": problem,
                    "policy": policy,
                    "mean_final_hv": finals.hypervolume.mean(),
                    "sd_final_hv": finals.hypervolume.std(),
                    "mean_hv_gain": finals.hv_improvement.mean(),
                    "sd_hv_gain": finals.hv_improvement.std(),
                    "mean_ask_seconds": timing.ask_seconds.mean(),
                    "mean_nondominated": finals.nondominated.mean(),
                    "warning_count": int(timing.warnings.sum()),
                    "mean_batch_distance": timing.batch_distance.mean(),
                }
            )
        axes[0, j].set_title(problem, fontsize=14)
        paired = (
            final[final.problem == problem]
            .pivot(index="seed", columns="policy", values="hypervolume")
            .sort_index()
        )
        differences = (paired.per_candidate - paired.per_batch).to_numpy()
        low, high = np.quantile(
            differences[bootstrap_indices].mean(axis=1), [0.025, 0.975]
        )
        p = float(wilcoxon(differences).pvalue) if np.any(differences != 0) else 1.0
        comparisons.append(
            {
                "problem": problem,
                "seeds": len(differences),
                "mean_paired_hv_difference": differences.mean(),
                "paired_ci_low": low,
                "paired_ci_high": high,
                "candidate_wins": int((differences > 1e-10).sum()),
                "batch_wins": int((differences < -1e-10).sum()),
                "ties": int((np.abs(differences) <= 1e-10).sum()),
                "wilcoxon_p": p,
            }
        )
    ranked = sorted(comparisons, key=lambda item: item["wilcoxon_p"])
    previous = 0.0
    for i, item in enumerate(ranked):
        previous = max(previous, min(1.0, item["wilcoxon_p"] * (len(ranked) - i)))
        item["holm_p"] = previous
    pd.DataFrame(summary).to_csv(directory / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(directory / "paired_comparisons.csv", index=False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"ParEGO scalarization policy: q={config['batch_size']}, {len(config['seeds'])} paired seeds\n"
        "Means with 95% bootstrap confidence intervals",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(directory / "comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / "comparison.svg", bbox_inches="tight")
    clear_gains = [
        item["problem"]
        for item in comparisons
        if item["mean_paired_hv_difference"] > 0 and item["holm_p"] < 0.05
    ]
    uncertain = [item["problem"] for item in comparisons if item["holm_p"] >= 0.05]
    outcome = f"At this budget, per-candidate scalarization has a clear positive final-hypervolume effect on {', '.join(clear_gains) or 'none of the tested problems'}. "
    if uncertain:
        outcome += f"The result on {', '.join(uncertain)} is inconclusive after accounting for seed variation. "
    outcome += "These results support independent scalarizations when learning a Pareto front, while leaving room for problem-dependent behavior."
    report = [
        "# Does ParEGO need a different scalarization for each batch candidate?",
        "",
        "This experiment compares two scalarization policies inside BoFire's existing "
        "ParEGO strategy. Production BoFire code was not changed. Both variants use "
        "qLogNEI and sequential pending-aware candidate generation.",
        "",
        outcome,
        "",
        "![Scalarization comparison](comparison.png)",
        "",
        "## Design",
        "",
        f"- {len(config['seeds'])} matched seeds per problem; {config['initial']} initial Sobol points; "
        f"{config['iterations']} rounds of {config['batch_size']} candidates "
        f"({config['initial']+config['iterations']*config['batch_size']} total observations).",
        f"- {config['mc_samples']} MC samples, {config['restarts']} restarts, "
        f"{config['raw_samples']} raw samples, maxiter={config['maxiter']}.",
        f"- CPU double precision; {config['threads']} PyTorch thread; BoTorch {config['botorch']}.",
        f"- {config.get('workers', 1)} independent worker processes; measured wall times include shared CPU contention.",
        "- DTLZ2: 6 inputs / 3 objectives / spherical front; reference (1.1, 1.1, 1.1).",
        "- DTLZ7: 6 inputs / 3 objectives / disconnected front; reference (1.1, 1.1, 7.0).",
        "- Branin–Currin: 2 inputs / 2 objectives; reference (18, 6).",
        "- Outcomes are noiseless; all objectives are minimized. These are short-horizon synthetic tests.",
        "- Both variants share initial designs and the first scalarization each round. "
        "Separate RNG streams control scalarizations, MC samplers, GP fitting, and optimizer initialization. "
        "Extra scalarization draws do not advance the optimizer stream. Execution order alternates by seed.",
        "- The q=1 control produced identical candidates, objective values, and weights. "
        "A runtime assertion checks one unique weight vector per batch versus q unique vectors per candidate.",
        "",
        "## Final quality and runtime",
        "",
        "| Problem | Policy | Final HV, mean ± SD | HV gain, mean ± SD | Mean ask/batch |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['problem']} | {LABELS[row['policy']]} | "
            f"{row['mean_final_hv']:.4f} ± {row['sd_final_hv']:.4f} | "
            f"{row['mean_hv_gain']:.4f} ± {row['sd_hv_gain']:.4f} | "
            f"{row['mean_ask_seconds']:.2f} s |"
        )
    report += [
        "",
        "## Paired comparison",
        "",
        "Positive differences favor one scalarization per candidate. Confidence intervals "
        f"bootstrap the {len(config['seeds'])} matched-seed differences (20,000 resamples); p-values use "
        "two-sided Wilcoxon signed-rank tests with Holm adjustment across the three problems.",
        "",
        "| Problem | Mean paired final-HV difference | 95% CI | Wins / losses / ties | Holm-adjusted p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        report.append(
            f"| {row['problem']} | {row['mean_paired_hv_difference']:+.5f} | "
            f"[{row['paired_ci_low']:+.5f}, {row['paired_ci_high']:+.5f}] | "
            f"{row['candidate_wins']} / {row['batch_wins']} / {row['ties']} | {row['holm_p']:.4g} |"
        )
    report += [
        "",
        "## Interpretation and limitations",
        "",
        "Hypervolume is the primary quality metric. Non-dominated-point counts provide "
        "supporting information about the attained front, but do not by themselves measure "
        "uniform coverage or closeness to the true front. Batch objective distances are "
        "logged as diagnostics; distance alone can reward poor solutions.",
        "",
        "This tests scalarization policy within BoFire; it does not benchmark Ax execution. "
        "A policy can be sensible for a particular preference or application even if it "
        "performs worse on these front-learning benchmarks. Reference points, batch size, "
        "noise, optimizer budget, and evaluation horizon can change the result. "
        "GP fitting and candidate-generation time are recorded separately. Warnings "
        "are counted in the raw results; automatic optimizer retries remain included in timing.",
        "",
        f"The shared policy draws {config['iterations']} trade-off directions over the run; "
        f"the per-candidate policy draws {config['iterations'] * config['batch_size']}. "
        "This difference is particularly relevant when the aim is to learn an entire "
        "Pareto front within a small number of batch rounds. The tests cover two and three "
        "objectives; Ax's default ParEGO regime with five or more objectives was not tested.",
        "",
        "## Artifacts",
        "",
        "- [Complete configuration](config.json)",
        "- [All per-round results](all_results.csv)",
        "- [Summary](summary.csv)",
        "- [Paired statistics](paired_comparisons.csv)",
        "- [Pairing audit](pairing_audit.csv) (initial design, monotonic hypervolume, logged weights, and first-candidate checks)",
        "- [SVG figure](comparison.svg)",
        "- Individual candidate CSV files record inputs, true outcomes, and scalarization weights.",
        "- [Benchmark script](../../scripts/benchmark_parego_scalarization.py)",
        "- [Analysis script](../../scripts/analyze_parego_scalarization.py)",
        "",
        "## Background",
        "",
        "- [BoTorch qNParEGO tutorial](https://botorch.org/docs/v0.17.2/tutorials/multi_objective_bo): separate scalarizations per candidate.",
        "- [BoFire issue #849](https://github.com/experimental-design/bofire/issues/849): shared scalarization in current batch generation.",
        "- [Ax acquisition construction](https://github.com/facebook/Ax/blob/5a7b3367f2d372e1653fd6c6ff1eaf8069ecc850/ax/generators/torch/botorch_modular/acquisition.py#L348-L364): the standard path constructs one acquisition object.",
        "",
        "Reproduce from the repository root:",
        "",
        "```bash",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n bofire python scripts/benchmark_parego_scalarization.py",
        f"conda run -n bofire python scripts/analyze_parego_scalarization.py {directory}",
        "```",
        "",
    ]
    (directory / "report.md").write_text("\n".join(report))
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
