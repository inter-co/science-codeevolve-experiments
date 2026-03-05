# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Stable perturbation strategy for Guided Local Search in Flow Shop Scheduling.
    
    Implements a robust approach that:
    1. Identifies machine bottlenecks using percentile-based analysis
    2. Applies conservative perturbations to avoid infeasibility
    3. Selects diverse jobs based on processing characteristics
    4. Maintains numerical stability and computational efficiency

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
    
    # Strategy 1: Identify bottleneck machines using percentile analysis
    machine_utilization = np.sum(time_matrix, axis=0)
    bottleneck_threshold = np.percentile(machine_utilization, 70)
    bottleneck_machines = np.where(machine_utilization > bottleneck_threshold)[0]
    
    # Apply conservative perturbations to jobs on bottleneck machines
    # Use modest increases to avoid extreme infeasibility
    if len(bottleneck_machines) > 0:
        for machine in bottleneck_machines:
            for job_idx in range(n):
                if time_matrix[job_idx, machine] > 0:
                    # Apply small, controlled increase (5-10%)
                    increase_factor = 1.05 + 0.05 * np.random.random()
                    new_matrix[job_idx, machine] *= increase_factor
    
    # Strategy 2: Job importance scoring based on processing time characteristics
    # Jobs with high variance across machines are more sensitive to changes
    job_variances = np.var(time_matrix, axis=1)
    job_totals = np.sum(time_matrix, axis=1)
    
    # Create importance scores with balanced weighting
    importance_scores = 0.6 * job_variances + 0.4 * job_totals
    
    # Strategy 3: Select diverse jobs for perturbation
    # Ensure we select at least 2 jobs, up to 5
    num_perturb_jobs = min(5, max(2, 2 + np.random.poisson(1)))
    
    # Sort by importance and select top jobs
    sorted_indices = np.argsort(-importance_scores)
    
    # Select jobs ensuring diversity and minimum count
    selected_indices = set()
    
    # Add top jobs (but cap at reasonable number)
    top_jobs_count = min(3, num_perturb_jobs)
    for i in range(top_jobs_count):
        if i < len(sorted_indices):
            selected_indices.add(sorted_indices[i])
    
    # Fill remaining slots with random jobs ensuring uniqueness
    while len(selected_indices) < num_perturb_jobs:
        job_idx = np.random.randint(0, n)
        selected_indices.add(job_idx)
    
    # Convert to actual job IDs in current sequence
    perturb_jobs = [current_sequence[job_idx] for job_idx in list(selected_indices)[:5]]
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
