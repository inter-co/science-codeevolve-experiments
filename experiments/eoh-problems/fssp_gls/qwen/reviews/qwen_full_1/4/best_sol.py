# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

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
    
    # Simplified but effective approach inspired by successful strategies:
    # 1. Focus on jobs with extreme processing times (both very long and very short)
    # 2. Include jobs from critical positions in the sequence
    # 3. Ensure diversity and sufficient number of jobs
    
    # Calculate total processing time per job
    job_totals = np.sum(time_matrix, axis=1)
    
    # Find jobs with extreme processing times - these are likely bottlenecks
    sorted_jobs_by_total = np.argsort(job_totals)
    
    # Select jobs with very high and very low total processing times
    perturb_jobs = set()
    
    # Always include the extremes
    if n >= 1:
        perturb_jobs.add(sorted_jobs_by_total[0])      # shortest total
        perturb_jobs.add(sorted_jobs_by_total[-1])     # longest total
    
    # Add some more extreme jobs if we have enough
    if n >= 4:
        perturb_jobs.add(sorted_jobs_by_total[n//4])   # quarter point
        perturb_jobs.add(sorted_jobs_by_total[3*n//4]) # three-quarter point
    
    # Add jobs from middle positions for swap impact (positions 1/4 to 3/4)
    middle_positions = []
    if n >= 4:
        start_pos = n // 4
        end_pos = 3 * n // 4
        middle_positions = list(range(start_pos, end_pos))
    
    # Sample a few jobs from middle positions for diversity
    if len(middle_positions) >= 2:
        np.random.shuffle(middle_positions)
        middle_sample = middle_positions[:min(2, len(middle_positions))]
        for pos in middle_sample:
            perturb_jobs.add(pos)
    
    # Convert to list and ensure at least 2 jobs
    perturb_jobs = list(perturb_jobs)
    
    # If we still don't have enough jobs, add some from the sequence
    if len(perturb_jobs) < 2:
        # Add jobs from the beginning of the sequence to ensure variety
        for i in range(len(perturb_jobs), min(2, n)):
            perturb_jobs.append(i)
    
    # Limit to maximum 5 jobs and ensure uniqueness
    perturb_jobs = list(dict.fromkeys(perturb_jobs))[:5]
    
    # Convert indices to actual job numbers in current sequence
    final_perturb_jobs = [current_sequence[i] for i in perturb_jobs[:5]]
    
    # Apply perturbation by slightly increasing processing times of selected jobs
    penalty_factor = 0.15  # Moderate penalty to encourage exploration
    
    for job_idx in final_perturb_jobs:
        # Increase processing times slightly for selected jobs
        new_matrix[job_idx] = time_matrix[job_idx] * (1.0 + penalty_factor)
    
    return new_matrix, final_perturb_jobs
# EVOLVE-BLOCK-END
