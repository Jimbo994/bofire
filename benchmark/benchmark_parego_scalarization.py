"""Paired BoFire experiment: one ParEGO scalarization per batch or candidate.

Production strategy code is unchanged. Both experimental variants inherit its
objective transformation and use the same sequential pending-aware optimizer.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import get_acquisition_function
from botorch.test_functions.multi_objective import DTLZ2, DTLZ7, BraninCurrin
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

import bofire
import bofire.strategies.predictives.qparego as parego_module
from bofire.data_models.acquisition_functions.api import qLogNEI
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.data_models.strategies.api import BotorchOptimizer, QparegoStrategy
from bofire.strategies.predictives.qparego import QparegoStrategy as RuntimeParEGO
from bofire.utils.torch_tools import tkwargs


def seed_for(seed: int, iteration: int, phase: int, index: int = 0) -> int:
    return int(
        np.random.SeedSequence([seed, iteration, phase, index]).generate_state(1)[0]
    )


class ExperimentalParEGO(RuntimeParEGO):
    def __init__(self, data_model, policy: str):
        super().__init__(data_model)
        self.policy = policy
        self.iteration = 0
        self.weight_log: list[list[float]] = []

    def _draw_objective(self, index: int):
        draw_seed = seed_for(self.seed, self.iteration, 1, index)
        original = parego_module.sample_simplex
        draws = []

        def record(*args, **kwargs):
            weights = original(*args, **kwargs)
            draws.append(weights.detach().cpu().reshape(-1).tolist())
            return weights

        with (
            torch.random.fork_rng(),
            patch.object(parego_module, "sample_simplex", record),
        ):
            torch.manual_seed(draw_seed)
            result = self._get_objective_and_constraints()
        assert len(draws) == 1
        return result, draws[0]

    def _get_acqfs(self, n: int):
        X_train, X_pending = self.get_acqf_input_tensors()
        shared, shared_weights = self._draw_objective(0)
        self.weight_log = []
        acqfs = []
        for i in range(n):
            (objective, constraints, etas), weights = (
                (shared, shared_weights)
                if self.policy == "per_batch" or i == 0
                else self._draw_objective(i)
            )
            self.weight_log.append(weights)
            # Separate RNG streams prevent additional scalarization draws from
            # changing MC sampler seeds or optimizer initial conditions.
            with torch.random.fork_rng():
                torch.manual_seed(seed_for(self.seed, self.iteration, 2, i))
                acqfs.append(
                    get_acquisition_function(
                        acquisition_function_name="qLogNEI",
                        model=self.model,
                        objective=objective,
                        X_observed=X_train,
                        X_pending=X_pending if i == 0 else None,
                        constraints=constraints,
                        eta=torch.as_tensor(etas, **tkwargs),
                        mc_samples=self.acquisition_function.n_mc_samples,
                        prune_baseline=True,
                        cache_root=None,
                        seed=seed_for(self.seed, self.iteration, 2, i),
                    )
                )
        assert len({tuple(w) for w in self.weight_log}) == (
            1 if self.policy == "per_batch" else n
        )
        return acqfs


def problem_spec(name: str):
    if name == "DTLZ2":
        problem = DTLZ2(dim=6, num_objectives=3)
        ref = [1.1, 1.1, 1.1]
    elif name == "DTLZ7":
        problem = DTLZ7(dim=6, num_objectives=3)
        # Tight objective-specific reference; avoids the uninformative 15^3
        # box in the generic DTLZ7 test-function default.
        ref = [1.1, 1.1, 7.0]
    else:
        problem = BraninCurrin()
        ref = [18.0, 6.0]
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key=f"x{i}", bounds=pair.tolist())
                for i, pair in enumerate(problem.bounds.T)
            ]
        ),
        outputs=Outputs(
            features=[
                ContinuousOutput(key=f"f{i}", objective=MinimizeObjective())
                for i in range(len(ref))
            ]
        ),
    )
    return problem, domain, np.asarray(ref)


def observe(problem, domain, candidates):
    X = torch.tensor(
        candidates[domain.inputs.get_keys()].to_numpy(copy=True), **tkwargs
    )
    with torch.no_grad():
        Y = problem(X).cpu().numpy()
    output = candidates[domain.inputs.get_keys()].reset_index(drop=True).copy()
    for i, key in enumerate(domain.outputs.get_keys()):
        output[key] = Y[:, i]
        output[f"valid_{key}"] = 1
    return output


def metrics(experiments, domain, reference):
    Y = torch.tensor(
        experiments[domain.outputs.get_keys()].to_numpy(copy=True), **tkwargs
    )
    hv = Hypervolume(ref_point=-torch.tensor(reference, **tkwargs)).compute(-Y)
    return float(hv), int(is_non_dominated(-Y).sum())


def run_one(args, problem_name, seed, policy, initial):
    problem, domain, reference = problem_spec(problem_name)
    spec = QparegoStrategy(
        domain=domain,
        acquisition_function=qLogNEI(n_mc_samples=args.mc_samples),
        acquisition_optimizer=BotorchOptimizer(
            n_restarts=args.restarts,
            n_raw_samples=args.raw_samples,
            maxiter=args.maxiter,
        ),
        seed=seed,
    )
    strategy = ExperimentalParEGO(spec, policy)
    torch.manual_seed(seed_for(seed, 0, 3))
    start = time.perf_counter()
    strategy.tell(initial.copy(deep=True))
    initial_fit = time.perf_counter() - start
    initial_hv, nd_count = metrics(strategy.experiments, domain, reference)
    rows = [
        {
            "problem": problem_name,
            "seed": seed,
            "policy": policy,
            "iteration": 0,
            "n_evaluations": len(initial),
            "hypervolume": initial_hv,
            "hv_improvement": 0.0,
            "nondominated": nd_count,
            "ask_seconds": 0.0,
            "fit_seconds": initial_fit,
            "warnings": 0,
            "batch_distance": np.nan,
        }
    ]
    observations = []
    for iteration in range(1, args.iterations + 1):
        strategy.iteration = iteration
        torch.manual_seed(seed_for(seed, iteration, 4))
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            candidates = strategy.ask(args.batch_size)
        ask_seconds = time.perf_counter() - start
        obs = observe(problem, domain, candidates)
        Y = obs[domain.outputs.get_keys()].to_numpy() / reference
        distance = float(torch.pdist(torch.tensor(Y)).mean())
        for i, record in enumerate(obs.to_dict("records")):
            observations.append(
                dict(
                    problem=problem_name,
                    seed=seed,
                    policy=policy,
                    iteration=iteration,
                    candidate=i,
                    weights=json.dumps(strategy.weight_log[i]),
                    **record,
                )
            )
        torch.manual_seed(seed_for(seed, iteration, 3))
        start = time.perf_counter()
        strategy.tell(obs)
        fit_seconds = time.perf_counter() - start
        hv, nd_count = metrics(strategy.experiments, domain, reference)
        rows.append(
            {
                "problem": problem_name,
                "seed": seed,
                "policy": policy,
                "iteration": iteration,
                "n_evaluations": len(strategy.experiments),
                "hypervolume": hv,
                "hv_improvement": hv - initial_hv,
                "nondominated": nd_count,
                "ask_seconds": ask_seconds,
                "fit_seconds": fit_seconds,
                "warnings": len(captured),
                "batch_distance": distance,
            }
        )
    return rows, observations


def run_pair(args, problem_name, seed):
    torch.set_num_threads(args.threads)
    problem, domain, _ = problem_spec(problem_name)
    X = domain.inputs.sample(args.initial, method=SamplingMethodEnum.SOBOL, seed=seed)
    initial = observe(problem, domain, X)
    policies = ["per_batch", "per_candidate"]
    if seed % 2:
        policies.reverse()
    for policy in policies:
        destination = args.output / f"{problem_name}_{seed}_{policy}.csv"
        if destination.exists():
            continue
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows, observations = run_one(args, problem_name, seed, policy, initial)
        pd.DataFrame(observations).to_csv(
            destination.with_suffix(".candidates.csv"), index=False
        )
        pd.DataFrame(rows).to_csv(destination, index=False)
        print(
            f"{problem_name} seed={seed:02} {policy:13} "
            f"HV={rows[-1]['hypervolume']:.5f} "
            f"total={time.perf_counter()-start:.1f}s",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems", nargs="+", default=["DTLZ2", "DTLZ7", "BraninCurrin"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--initial", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mc-samples", type=int, default=128)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--raw-samples", type=int, default=128)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/scalarization_paired")
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "output": str(args.output),
        "botorch": botorch.__version__,
        "bofire": bofire.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "dtype": str(tkwargs["dtype"]),
        "note": "All outcomes noiseless; CPU; per-task output supports resume.",
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output configuration differs; use a new directory.")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_pair, args, name, seed)
            for seed in args.seeds
            for name in args.problems
        ]
        for future in as_completed(futures):
            future.result()
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
