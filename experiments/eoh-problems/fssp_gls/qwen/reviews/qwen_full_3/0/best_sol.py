# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Improved perturbation strategy for Guided Local Search in Flow Shop Scheduling.
    Based on insights from inspiration programs but with more stable implementation.
    
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
    
    # Calculate processing time variance for each job - jobs with higher variance 
    # are more sensitive to perturbations
    job_variances = np.var(time_matrix, axis=1)
    
    # Create penalty weights: jobs with higher variance get higher penalties
    max_variance = np.max(job_variances)
    penalty_weights = np.ones(n)
    if max_variance > 0:
        penalty_weights = 1.0 + 0.6 * (job_variances / max_variance)
    
    # Apply penalties to the time matrix - increase processing times of sensitive jobs
    # Use a moderate penalty to avoid over-penalizing while still providing perturbation
    penalty_factor = 0.15  # Fixed penalty factor for stability
    for i in range(n):
        # Clamp the penalty factor to prevent extreme values
        clamped_penalty = min(penalty_weights[i] * (1.0 + np.random.uniform(0.05, 0.15)), 3.0)
        new_matrix[i] = time_matrix[i] * clamped_penalty
    
    # Select jobs to perturb based on a weighted score:
    # - Higher variance jobs (more sensitive)
    # - Higher total processing time jobs (more impactful)
    processing_times = np.sum(time_matrix, axis=1)
    normalized_processing = processing_times / np.max(processing_times) if np.max(processing_times) > 0 else processing_times
    
    # Score based on variance and processing time with random component for exploration
    scores = (
        0.6 * (job_variances / np.max(job_variances) if np.max(job_variances) > 0 else np.zeros(n)) +
        0.3 * normalized_processing +
        0.1 * np.random.random(n)
    )
    
    # Select top jobs (between 2 and 5)
    num_selected = max(2, min(5, int(n * 0.2)))  # 20% of jobs, between 2-5
    selected_indices = np.argsort(-scores)[:num_selected]
    
    # Convert to list and ensure minimum of 2 jobs
    selected_jobs = list(selected_indices)
    
    # If we have less than 2 jobs, add some more based on processing time
    if len(selected_jobs) < 2:
        remaining_jobs = list(np.argsort(-processing_times)[:5])
        for job in remaining_jobs:
            if job not in selected_jobs:
                selected_jobs.append(job)
                if len(selected_jobs) >= 2:
                    break
    
    # Cap at 5 jobs maximum
    selected_jobs = selected_jobs[:5]
    
    return new_matrix, selected_jobs
# EVOLVE-BLOCK-END
