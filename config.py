# ============================================================
# Configuration Parameters
# ============================================================

import random
import numpy as np

# Problem size (SCALABLE PARAMETERS)
N_VARS = 10          # number of variables
DOMAIN = 10          # domain size of each variable
N_SAMPLES = 100_000  # number of training examples
SEED = 42

# Initialize random seeds
random.seed(SEED)
np.random.seed(SEED)
