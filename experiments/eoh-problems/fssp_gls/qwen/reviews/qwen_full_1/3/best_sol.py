# EVOLVE-BLOCK-START

import numpy as np
import random

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation uses a streamlined approach:
    1. Identifies bottleneck machines efficiently
    2. Prioritizes jobs based on processing time and machine load impact
    3. Uses simple stochastic selection for good diversity
    4. Applies precise perturbation to encourage exploration

    Args:
        current_sequence: list of job indices in the current sequence.
        time_matrix: np.ndarray of shape (n, m) - execution time of each job
            on each machine.
        m: int - number of machines.
        n: int - number of jobs.

    Returns:
        new_matrix: np.ndarray of shape (n, m) - updated execution time matrix.
        perturb_jobs: list of job indices to perturb (length > 1, max 5).
    """
    new_matrix = time_matrix.copy()
    
    # Step 1: Identify bottleneck machines efficiently
    machine_loads = np.sum(time_matrix, axis=0)
    max_load = np.max(machine_loads)
    
    penalty_factors = np.ones(m)
    if max_load > 1e-8:
        penalty_factors = 1.0 + 0.25 * (machine_loads / max_load)
        penalty_factors = np.clip(penalty_factors, 1.0, 2.0)
    
    # Step 2: Calculate job importance scores using vectorized operations for efficiency
    job_processing_times = np.sum(time_matrix, axis=1)
    
    # Vectorized computation of load impact for all jobs
    total_times = np.sum(time_matrix, axis=1, keepdims=True)
    # Avoid division by zero
    total_times = np.where(total_times == 0, 1, total_times)
    
    # Compute load impact for each job on each machine, then sum
    load_impacts = np.sum(time_matrix * penalty_factors, axis=1) / total_times.flatten()
    
    # Normalize processing times
    max_proc = np.max(job_processing_times)
    proc_importance = job_processing_times / (max_proc + 1e-8)
    
    # Combine importance scores
    importance_scores = (
        0.5 * proc_importance +      # Processing time importance
        0.5 * np.clip(load_impacts, 0, 1)  # Load impact importance (normalized)
    )
    
    # Step 3: Simple but effective stochastic selection
    # Use softmax with temperature for selection
    temp = 1.2  # Lower temperature for more focused selection
    selection_probs = np.exp(importance_scores * temp)
    selection_probs = selection_probs / np.sum(selection_probs)
    
    # Select up to 5 distinct jobs using direct sampling
    perturb_jobs = []
    selected_indices = set()
    
    # Sample up to 5 jobs ensuring uniqueness
    attempts = 0
    max_attempts = n * 2
    
    while len(perturb_jobs) < 5 and attempts < max_attempts:
        chosen_job = np.random.choice(n, p=selection_probs)
        if chosen_job not in selected_indices:
            perturb_jobs.append(chosen_job)
            selected_indices.add(chosen_job)
        attempts += 1
    
    # Ensure at least 2 jobs (required by constraints)
    if len(perturb_jobs) < 2:
        # Fill with high processing time jobs
        high_proc_indices = np.argsort(-job_processing_times)
        for idx in high_proc_indices:
            if idx not in selected_indices and len(perturb_jobs) < 5:
                perturb_jobs.append(idx)
                selected_indices.add(idx)
    
    # Step 4: Apply perturbation with more aggressive but controlled changes
    for job_idx in perturb_jobs:
        # More aggressive perturbation to encourage better exploration
        perturbation_factor = 0.9 + 0.4 * random.random()  # Range [0.9, 1.3]
        new_matrix[job_idx] = time_matrix[job_idx] * perturbation_factor
    
    # Final cleanup to ensure correct output format
    perturb_jobs = perturb_jobs[:5]
    if len(perturb_jobs) < 2:
        remaining_jobs = [i for i in range(n) if i not in perturb_jobs]
        for job in remaining_jobs[:2-len(perturb_jobs)]:
            perturb_jobs.append(job)
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
