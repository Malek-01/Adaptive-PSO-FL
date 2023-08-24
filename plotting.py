import numpy as np
import matplotlib.pyplot as plt
import params

from algorithms import CPSOAlgorithm, HCLPSOAlgorithm, SPSOAlgorithm, EPSOAlgorithm
from pyswarms_gbest.gbest import optimize

cpso = np.zeros((params.num_iterations, params.num_runs))
spso = np.zeros((params.num_iterations, params.num_runs))
hcpso = np.zeros((params.num_iterations, params.num_runs))
epso = np.zeros((params.num_iterations, params.num_runs))
import math

def plot(func_bench): 
    for run in range(params.num_runs):
        cpso[:,run] = CPSOAlgorithm(run, func_bench)[:,run]
        spso[:,run] = SPSOAlgorithm(run, func_bench)[:,run]
        hcpso[:,run] = HCLPSOAlgorithm(run, func_bench)[:,run]
        epso[:,run] = EPSOAlgorithm(run, func_bench)[:,run]
    
    # Calculate the mean best fitness value across runs at each iteration
    # and the gbest across at each run
    mean_it_CPSO, min_run_CPSO = np.mean(cpso, axis=1), np.min(cpso, axis=0)
    mean_it_SPSO, min_run_SPSO = np.mean(spso, axis=1), np.min(spso, axis=0)
    mean_it_HCPSO, min_run_HCPSO = np.mean(hcpso, axis=1), np.min(hcpso, axis=0)
    mean_it_EPSO, min_run_EPSO = np.mean(epso, axis=1), np.min(epso, axis=0)
    mean_it_CPSO = [math.log(x) for x in mean_it_CPSO] 
    mean_it_SPSO= [math.log(x) for x in mean_it_SPSO]
    mean_it_HCPSO = [math.log(x) for x in mean_it_HCPSO]
    mean_it_EPSO = [math.log(x) for x in mean_it_EPSO]
    h1, h2, h3 = optimize(func_bench)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(mean_it_CPSO, label='CPSO')
    plt.plot(mean_it_SPSO, label='SPSO')
    plt.plot(mean_it_HCPSO, label='HCPSO')
    plt.plot(mean_it_EPSO, label='EPSO')
    plt.plot(h1, label='PSO', linewidth=2)
    plt.plot(h2, label='PSO-R-ML', linewidth=2)
    plt.plot(h3, label='PSO-R-D', linewidth=2)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Objective Value (mean)", fontsize=14)
    plt.title("Convergence Comparison of PSO Variants on "+func_bench.__name__+" Function", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.yscale('log')  # Using a logarithmic scale for better visualization of convergence
    plt.tight_layout()
    plt.show()

    fac_CPSO = np.min(min_run_CPSO), np.mean(min_run_CPSO), np.std(min_run_CPSO)
    fac_SPSO = np.min(min_run_SPSO), np.mean(min_run_SPSO), np.std(min_run_SPSO)
    fac_HCPSO = np.min(min_run_HCPSO), np.mean(min_run_HCPSO), np.std(min_run_HCPSO)
    fac_EPSO = np.min(min_run_EPSO), np.mean(min_run_EPSO), np.std(min_run_EPSO)

    return fac_CPSO, fac_SPSO, fac_HCPSO, fac_EPSO