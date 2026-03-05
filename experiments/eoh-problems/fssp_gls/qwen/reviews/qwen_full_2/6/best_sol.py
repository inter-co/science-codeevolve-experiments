# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation focuses on jobs with high impact on makespan by:
    1. Identifying jobs near the end of the sequence (highest impact)
    2. Applying strategic time modifications to encourage resequencing
    3. Ensuring sufficient exploration while maintaining efficiency

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
    
    # Strategy 1: Focus on jobs that have high impact on makespan
    # Jobs near the end of the sequence typically have the highest impact
    # We target the last 20% of jobs in the current sequence
    end_positions = max(2, int(n * 0.2))
    perturb_jobs = []
    
    # Add jobs from the end of the current sequence (most impactful)
    for i in range(min(end_positions, len(current_sequence))):
        job_idx = current_sequence[-(i+1)]
        if job_idx not in perturb_jobs:
            perturb_jobs.append(job_idx)
    
    # Strategy 2: Add some diversity by including jobs from the middle
    # This helps avoid getting stuck in local optima too quickly
    mid_positions = max(1, int(n * 0.1))  # Add 10% from middle
    for i in range(mid_positions):
        idx = len(current_sequence) // 2 + i
        if idx < len(current_sequence):
            job_idx = current_sequence[idx]
            if job_idx not in perturb_jobs:
                perturb_jobs.append(job_idx)
    
    # Strategy 3: Ensure we have at least 2 jobs, add from beginning if needed
    if len(perturb_jobs) < 2:
        for i in range(2 - len(perturb_jobs)):
            if i < len(current_sequence):
                job_idx = current_sequence[i]
                if job_idx not in perturb_jobs:
                    perturb_jobs.append(job_idx)
    
    # Limit to max 5 jobs
    perturb_jobs = perturb_jobs[:5]
    
    # Strategy 4: Apply time modifications to selected jobs
    # Increase processing times to encourage exploration without drastic changes
    for job_idx in perturb_jobs:
        # Apply a small but meaningful perturbation (5-15% change)
        # Use a slight random variation to create diversity
        perturbation_factor = 1.05 + 0.1 * np.random.random()  # Between 1.05 and 1.15
        new_matrix[job_idx] = time_matrix[job_idx] * perturbation_factor
    
    # Final safeguard to ensure we have at least 2 jobs
    if len(perturb_jobs) < 2:
        # Add first few jobs from current sequence to guarantee minimum count
        for i in range(2 - len(perturb_jobs)):
            if i < len(current_sequence):
                job_idx = current_sequence[i]
                if job_idx not in perturb_jobs:
                    perturb_jobs.append(job_idx)
    
    # Ensure no duplicates and limit to 5
    perturb_jobs = list(dict.fromkeys(perturb_jobs))[:5]
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
