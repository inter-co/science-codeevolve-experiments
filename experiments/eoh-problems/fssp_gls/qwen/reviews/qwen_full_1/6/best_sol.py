# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using variance-based and bottleneck-aware strategies for GLS in FSSP.

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
    
    # Strategy 1: Compute processing time variance for each job across machines
    job_variances = np.var(time_matrix, axis=1)
    
    # Strategy 2: Identify bottleneck machines based on load
    machine_loads = np.sum(time_matrix, axis=0)
    bottleneck_machines = np.argsort(machine_loads)[-min(3, m):]
    
    # Strategy 3: Create job importance scores based on variance and bottleneck interaction
    job_importance_scores = np.zeros(n)
    
    for i in range(n):
        job_idx = current_sequence[i]
        # Base score from processing time variance
        job_importance_scores[i] = job_variances[job_idx]
        
        # Additional bonus for jobs that process on bottleneck machines
        bottleneck_bonus = 0
        for machine_idx in bottleneck_machines:
            if time_matrix[job_idx, machine_idx] > 0:
                bottleneck_bonus += time_matrix[job_idx, machine_idx]
        job_importance_scores[i] += 0.3 * bottleneck_bonus
    
    # Strategy 4: Perturbation Matrix - Apply small random perturbations to all jobs
    # This helps escape local optima without breaking feasibility
    perturbation_strength = 0.12  # 12% perturbation (slightly less aggressive)
    for i in range(n):
        job_idx = current_sequence[i]
        # Apply small random multiplicative perturbation
        perturbation = 1.0 + np.random.uniform(-perturbation_strength, perturbation_strength)
        new_matrix[job_idx] = time_matrix[job_idx] * perturbation
    
    # Strategy 5: Job Selection Strategy - Simple and effective
    # Select top 3-5 jobs with highest importance scores
    sorted_indices = np.argsort(-job_importance_scores)
    
    # Select 3-5 jobs (ensure at least 2)
    # Using a deterministic approach for better reproducibility
    num_perturb_jobs = 3 + (len(current_sequence) % 3)  # Deterministic but varied
    num_perturb_jobs = min(max(2, num_perturb_jobs), 5)
    
    # Select top jobs by importance score
    selected_indices = sorted_indices[:num_perturb_jobs].tolist()
    
    # Convert indices back to actual job IDs
    perturb_jobs = [current_sequence[i] for i in selected_indices]
    
    # Ensure we don't exceed 5 jobs and always have at least 2
    perturb_jobs = perturb_jobs[:5]
    if len(perturb_jobs) < 2:
        # Fallback: add the highest variance jobs to ensure minimum count
        sorted_by_variance = np.argsort(-job_variances)
        for i in range(n):
            if len(perturb_jobs) >= 2:
                break
            job_id = current_sequence[sorted_by_variance[i]]
            if job_id not in perturb_jobs:
                perturb_jobs.append(job_id)
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
