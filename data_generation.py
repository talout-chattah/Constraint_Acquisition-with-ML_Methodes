# ============================================================
# Dataset Generation
# ============================================================

import numpy as np
from constraints import check_constraints


def generate_dataset(n_vars, domain, constraints, n_samples):
    """
    Generate a dataset by sampling random variable assignments
    and labeling them according to the constraints.
    """
    X = np.random.randint(
        1, domain + 1, size=(n_samples, n_vars)
    )
    y = np.array([
        check_constraints(x, constraints) for x in X
    ])
    return X, y


def sample_points(n_points, n_vars, domain):
    """
    Sample random points from the variable space.
    Used for distillation.
    """
    return np.random.randint(
        1, domain + 1, size=(n_points, n_vars)
    )
