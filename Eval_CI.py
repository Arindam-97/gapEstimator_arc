import numpy as np

def get_ci(bagEstimator, gapProblem, data, alpha, k, B):
    alpha_ = alpha/2
    bound, point_estimate, _ = bagEstimator.run(gapProblem, data, alpha_ , k, B)
    LB = bound
    UB = point_estimate+(point_estimate-bound)
    return (LB, UB)



def evaluate_ci_performance(bagEstimator, gapProblem, N, k, B, n_reps, target, alpha=0.05, seed=654321):
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

    rng = np.random.default_rng(seed)

    lengths = []
    covered = []

    for _ in range(n_reps):
        data = rng.normal(size=N)

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