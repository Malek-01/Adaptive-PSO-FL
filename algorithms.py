import numpy as np
import params

best_fitness_evolution = np.zeros((params.num_iterations, params.num_runs))
mean_gbest_evolution = np.zeros((params.num_iterations, params.num_runs))
alpha = 0.1  # Heterogeneity factor

particles_position = np.random.uniform(-600, 600, size=(params.num_particles, params.num_dimensions))
particles_velocity = np.zeros((params.num_particles, params.num_dimensions))
personal_best = particles_position.copy()

def compute_global_best(fac, func_bench):
    global_best_idx = np.argmin([func_bench(p) for p in personal_best])
    global_best = personal_best[global_best_idx].copy()
    for i in range(params.num_particles):
            r1 = np.random.rand(params.num_dimensions)
            r2 = np.random.rand(params.num_dimensions)   
            cognitive_component = params.c1 * r1 * (personal_best[i] - particles_position[i]) * fac
            social_component = params.c2 * r2 * (global_best - particles_position[i]+0) * fac
            particles_velocity[i] = params.w * particles_velocity[i] + cognitive_component + social_component
            particles_position[i] = particles_position[i] + particles_velocity[i]
        # Update personal best
            if func_bench(particles_position[i]) < func_bench(personal_best[i]):
                personal_best[i] = particles_position[i].copy()
                # Update global best if necessary
                if func_bench(personal_best[i]) < func_bench(global_best):
                    global_best = personal_best[i].copy()
    return global_best

# CPSO algorithm
def CPSOAlgorithm(run, func_bench):
    for iteration in range(params.num_iterations):
        global_best = compute_global_best(1, func_bench)
        best_fitness_evolution[iteration, run] = func_bench(global_best)
    return best_fitness_evolution

# HCLPSO algorithm
def HCLPSOAlgorithm(run, func_bench):
    global_best_idx = np.argmin([func_bench(p) for p in personal_best])
    global_best = personal_best[global_best_idx].copy()
    gbest_values = []
    heterogeneity_factor = np.random.uniform(1 - alpha, 1 + alpha, size=params.num_dimensions)
    for iteration in range(params.num_iterations):
        global_best = compute_global_best(heterogeneity_factor, func_bench)
        gbest_values.append(func_bench(global_best))
    mean_gbest_evolution[:, run] = gbest_values
    return mean_gbest_evolution

# SPSO algorithm
def SPSOAlgorithm(run, func_bench):
    global_best_idx = np.argmin([func_bench(p) for p in personal_best])
    global_best = personal_best[global_best_idx].copy()
    gbest_values = []
    for iteration in range(params.num_iterations):
        global_best = compute_global_best(1, func_bench)
        gbest_values.append(func_bench(global_best))
    mean_gbest_evolution[:, run] = gbest_values
    return mean_gbest_evolution

# EPSO algorithm
def EPSOAlgorithm(run, func_bench):
    global_best_idx = np.argmin([func_bench(p) for p in personal_best])
    global_best = personal_best[global_best_idx].copy()
    gbest_values = []
    # EPSO algorithm
    for iteration in range(params.num_iterations):
        global_best = compute_global_best(params.w, func_bench)
        gbest_values.append(func_bench(global_best))
    
    best_fitness_evolution[:, run] = gbest_values
    return best_fitness_evolution