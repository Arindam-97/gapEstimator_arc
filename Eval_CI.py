import numpy as np
import pandas as pd
from tqdm import tqdm
from GapEstimator import BagProcedure

def get_ci(bagEstimator, gapProblem, data, alpha, k, B):
    alpha_ = alpha/2
    bound, point_estimate, _ = bagEstimator.run(gapProblem, data, alpha_ , k, B)
    LB = bound
    UB = point_estimate+(point_estimate-bound)
    return (LB, UB)



def evaluate_ci_performance(bagEstimator, gapProblem, N, k, B, n_reps, target, rng, data_sampler, alpha=0.05):
    """
    Runs repeated simulations and returns average CI length and estimated coverage.

    Parameters
    ----------
    N : int
        Data size.
    k : int
        Resample size.
    B : int
        Bootstrap size.
    n_reps : int
        Number of Monte Carlo replications.
    target : float
        True target value to check coverage against.
    alpha : float
        Miscoverage level. Default is 0.05.
    seed : int
        Base random seed.

    Returns
    -------
    dict
        Average CI length and coverage estimate.
    """

    lengths = []
    covered = []

    for _ in range(n_reps):
        data = data_sampler(rng, N)

        lower, upper = get_ci(bagEstimator, gapProblem, data, alpha, k, B)

        lengths.append(upper - lower)
        covered.append(lower <= target<=upper)

    return {
        "avg_length": np.mean(lengths),
        "coverage": np.mean(covered),
        "n_reps": n_reps,
        "N": N,
        "k": k,
        "B": B,
        "alpha": alpha
    }



def run_grid_experiment(
    gapProblem,
    data_sampler,
    target,
    k=50,
    n_count=8,
    n_max=5000,
    B_mults=[0.5, 0.75, 1.0, 1.5, 2.0],
    n_reps=40,
    alpha=0.05,
    replace=True,
    debias=False,
    pointVar=False,
    seed=123456,
):
    """
    Run evaluate_ci_performance over a grid of N and B values.

    N runs from k to n_max using n_count grid points.
    For each N, B is set to ceil(B_mult * N).

    Parameters
    ----------
    gapProblem : GapProblem
        Problem object passed to evaluate_ci_performance.

    data_sampler : callable
        Function of the form data_sampler(rng, N), returning data of size N.

    target : float
        True target value for coverage.

    k : int
        Resample size.

    n_count : int
        Number of N values between k and n_max.

    n_max : int
        Maximum N value.

    B_mults : iterable
        Multipliers used to define B = ceil(B_mult * N).

    n_reps : int
        Number of Monte Carlo replications per grid point.

    alpha : float
        Miscoverage level.

    replace, debias, pointVar : bool
        Arguments passed to BagProcedure.

    seed : int
        Master random seed.

    Returns
    -------
    pd.DataFrame
        Grid results, one row per configuration.
    """

    N_grid = np.linspace(k, n_max, n_count).astype(int)
    seed_rng = np.random.default_rng(seed)

    rows = []

    total_runs = len(N_grid) * len(B_mults)

    for N in tqdm(N_grid, total=len(N_grid), desc="Running N grid"):
        for B_mult in B_mults:
            B = int(np.ceil(B_mult * N))

            data_seed = int(seed_rng.integers(0, 2**32 - 1))
            resample_seed = int(seed_rng.integers(0, 2**32 - 1))

            data_rng = np.random.default_rng(data_seed)
            resample_rng = np.random.default_rng(resample_seed)

            bagEstimator = BagProcedure(
                replace=replace,
                debias=debias,
                pointVar=pointVar,
                rng=resample_rng
            )

            results = evaluate_ci_performance(
                bagEstimator,
                gapProblem,
                N=N,
                k=k,
                B=B,
                n_reps=n_reps,
                target=target,
                rng=data_rng,
                data_sampler=data_sampler,
                alpha=alpha
            )

            row = results.copy()

            row.update({
                "B_mult": B_mult,
                "data_seed": data_seed,
                "resample_seed": resample_seed,
                "replace": replace,
                "debias": debias,
                "pointVar": pointVar,
            })

            rows.append(row)

    return pd.DataFrame(rows)
