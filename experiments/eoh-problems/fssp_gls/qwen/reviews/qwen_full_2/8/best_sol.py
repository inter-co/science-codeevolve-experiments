# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using a robust, efficient approach for GLS in FSSP.

    This implementation focuses on:
    - Simple but effective bottleneck detection
    - Clear job importance scoring based on multiple criteria
    - Deterministic selection with minimal randomness for consistency
    - Controlled perturbation that maintains schedule quality

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
    
    # Step 1: Identify bottleneck machines using simple load analysis
    machine_loads = np.sum(time_matrix, axis=0)
    avg_load = np.mean(machine_loads)
    bottleneck_machines = np.where(machine_loads > avg_load)[0]
    
    # Step 2: Calculate job importance scores
    job_variances = np.var(time_matrix, axis=1)
    job_totals = np.sum(time_matrix, axis=1)
    
    # Score based on processing time on bottleneck machines
    bottleneck_scores = np.zeros(n)
    for machine_idx in bottleneck_machines:
        bottleneck_scores += time_matrix[:, machine_idx]
    
    # Positional score (jobs later in sequence are more critical)
    positional_scores = np.arange(n, 0, -1)
    
    # Step 3: Combine scores with clear weights
    # This creates a stable, interpretable importance ranking
    importance_scores = (
        0.4 * job_variances +      # Uncertainty factor
        0.4 * bottleneck_scores +  # Bottleneck impact
        0.2 * job_totals +         # Overall processing time
        0.1 * positional_scores    # Sequence position
    )
    
    # Step 4: Select jobs for perturbation deterministically but with some randomness
    # Sort by importance and take top 3-5 jobs
    sorted_indices = np.argsort(-importance_scores)
    
    # Select 3-5 jobs with slight randomness for diversity
    num_jobs = min(5, max(3, 3 + np.random.randint(0, 3)))
    perturb_jobs = sorted_indices[:num_jobs].tolist()
    
    # Ensure we have at least 2 jobs
    if len(perturb_jobs) < 2:
        # Add high-total-time jobs to meet minimum requirement
        total_times = np.sum(time_matrix, axis=1)
        high_jobs = list(np.argsort(-total_times)[:5])
        for job in high_jobs:
            if len(perturb_jobs) >= 5:
                break
            if job not in perturb_jobs:
                perturb_jobs.append(job)
    
    # Limit to 5 jobs maximum
    perturb_jobs = perturb_jobs[:5]
    
    # Final safety check to ensure minimum jobs
    if len(perturb_jobs) < 2:
        total_times = np.sum(time_matrix, axis=1)
        perturb_jobs = list(np.argsort(-total_times)[:2])
    
    # Step 5: Apply controlled perturbation to selected jobs
    # Use a consistent, effective perturbation strategy
    for job_idx in perturb_jobs:
        # Determine perturbation intensity based on job characteristics
        job_variance = job_variances[job_idx]
        job_total = job_totals[job_idx]
        
        # Base perturbation strength varies by job category
        if job_variance > np.percentile(job_variances, 75):  # High variance
            strength = 0.25 + 0.1 * np.random.random()  # Aggressive
        elif job_total > np.percentile(job_totals, 75):  # Long job
            strength = 0.15 + 0.1 * np.random.random()  # Moderate
        else:  # Normal job
            strength = 0.10 + 0.05 * np.random.random()  # Conservative
        
        # Apply multiplicative perturbation with bounded changes
        for machine_idx in range(m):
            # Generate perturbation factor between 0.7 and 1.3
            perturbation_factor = 0.7 + 0.6 * np.random.random()
            
            # Apply bounded perturbation (±30% change)
            new_value = time_matrix[job_idx, machine_idx] * perturbation_factor
            new_value = np.clip(new_value, 0.7 * time_matrix[job_idx, machine_idx], 
                              1.3 * time_matrix[job_idx, machine_idx])
            
            new_matrix[job_idx, machine_idx] = new_value
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
