import numpy as np
import params

from pyswarms_gbest.global_best import GlobalBestPSO2
from pyswarms_gbest.Pyswarms_gbest import GlobalBestPSO
from pyswarms_gbest.Pyswarms_entropy_gbest import GlobalBestPSO_entropy

# Set up the optimizer
options = {'c1': params.c1, 'c2': params.c2, 'w': params.w}

# Define the bounds of the search space
lb = np.array([-600, -600])  # Lower bounds
ub = np.array([600, 600])    # Upper bounds

# Create a ParticleSwarmOptimization instance
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options, bounds=(lb, ub))
optimizerXY = GlobalBestPSO2(n_particles=200, dimensions=2, options=options, bounds=(lb, ub))
optimizerEntrpy = GlobalBestPSO_entropy(n_particles=150, dimensions=2, options=options, bounds=(lb, ub))

def optimize(func_bench):
# Perform the optimization
    h1 = optimizerXY.optimize(func_bench, iters=params.num_iterations)[2]
    h2 = optimizer.optimize(func_bench, iters=params.num_iterations)[2]
    h3 = optimizerEntrpy.optimize(func_bench, iters=params.num_iterations)[2]
    return h1, h2, h3
