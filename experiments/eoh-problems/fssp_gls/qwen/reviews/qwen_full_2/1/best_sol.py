# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation uses a simple yet effective perturbation strategy:
    1. Randomly select 2-5 jobs to perturb
    2. Modify their processing times by applying random multipliers
    3. This creates sufficient disruption to escape local optima

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
    
    # Simple random selection of jobs to perturb (2-5 jobs)
    num_perturb_jobs = np.random.randint(2, min(6, n + 1))
    perturb_jobs = np.random.choice(n, size=num_perturb_jobs, replace=False).tolist()
    
    # Apply perturbation to selected jobs
    # Use random multipliers between 0.7 and 1.3 to create disruption
    # This ensures jobs become slightly easier or harder to schedule
    for job_idx in perturb_jobs:
        # Apply random multiplier between 0.7 and 1.3
        multiplier = 0.7 + np.random.random() * 0.6
        new_matrix[job_idx] = time_matrix[job_idx] * multiplier
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
