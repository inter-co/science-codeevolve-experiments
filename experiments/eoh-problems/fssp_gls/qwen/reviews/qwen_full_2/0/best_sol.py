# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation uses a hybrid approach combining:
    1. Sequence-based job selection (focus on recent positions)
    2. Processing time-based perturbation (high variance jobs get more attention)
    3. Balanced randomization to ensure diversity

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
    
    # Strategy 1: Select jobs based on their position in the current sequence
    # Focus on jobs that have been recently scheduled (end of sequence)
    recent_positions = min(5, len(current_sequence) // 2)
    recent_jobs = []
    if recent_positions > 0:
        recent_jobs = current_sequence[-recent_positions:]
    
    # Strategy 2: Select jobs based on processing time variance (high variance = more sensitive)
    job_variances = np.var(time_matrix, axis=1)
    high_variance_jobs = np.argsort(-job_variances)[:min(3, n)]
    
    # Strategy 3: Combine selections with some randomness
    candidate_jobs = set(recent_jobs + list(high_variance_jobs))
    
    # Ensure we have at least 2 jobs
    if len(candidate_jobs) < 2:
        # Fill with random jobs
        remaining_jobs = set(range(n)) - candidate_jobs
        additional_count = 2 - len(candidate_jobs)
        if additional_count > 0 and remaining_jobs:
            additional_jobs = np.random.choice(list(remaining_jobs), size=min(additional_count, len(remaining_jobs)), replace=False)
            candidate_jobs.update(additional_jobs)
    
    # Convert to list and select final jobs
    candidate_list = list(candidate_jobs)
    num_perturb_jobs = min(5, max(2, len(candidate_list)))
    
    # If we have more candidates than needed, sample randomly
    if len(candidate_list) > num_perturb_jobs:
        perturb_jobs = np.random.choice(candidate_list, size=num_perturb_jobs, replace=False).tolist()
    else:
        perturb_jobs = candidate_list
    
    # Apply perturbation to selected jobs
    # Use multipliers that create meaningful changes without being extreme
    for job_idx in perturb_jobs:
        # Apply random multiplier between 0.8 and 1.2 to create moderate disruption
        multiplier = 0.8 + np.random.random() * 0.4
        new_matrix[job_idx] = time_matrix[job_idx] * multiplier
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
