# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation uses a novel multi-strategy perturbation approach:
    1. Identifies machine bottlenecks based on load distribution
    2. Exploits job sequencing relationships through conflict detection
    3. Applies strategic time modifications for enhanced exploration
    4. Ensures diverse and impactful job selection

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
    
    # Strategy 1: Machine-based bottleneck identification
    # Calculate machine loads and identify bottlenecks efficiently
    machine_loads = np.sum(time_matrix, axis=0)
    avg_load = np.mean(machine_loads)
    bottleneck_machines = np.where(machine_loads > avg_load * 1.2)[0]
    
    # Strategy 2: Job conflict detection using critical path analysis
    # Identify jobs that may cause delays due to sequencing conflicts
    conflict_scores = np.zeros(n)
    
    # Create job position mapping
    job_positions = np.array(current_sequence)
    
    # Vectorized approach to detect conflicts between adjacent jobs
    # For each job, check if it's followed by another job in sequence that might cause issues
    for i in range(n-1):
        job_i = job_positions[i]
        job_j = job_positions[i+1]
        
        # Score based on processing time differences that could cause delays
        diff = np.abs(time_matrix[job_i] - time_matrix[job_j])
        conflict_scores[job_i] += np.sum(diff) * 0.2
    
    # Strategy 3: Statistical-based job impact scoring
    # Calculate processing time variance for each job
    time_variances = np.var(time_matrix, axis=1)
    
    # Combine different scoring mechanisms efficiently
    # 1. Total processing time (higher = more critical)
    total_times = np.sum(time_matrix, axis=1)
    
    # 2. Variance in processing times (higher = more sensitive to changes)
    # 3. Bottleneck machine influence (jobs on overloaded machines)
    bottleneck_influence = np.zeros(n)
    for machine in bottleneck_machines:
        # Find jobs that use this bottleneck machine
        jobs_on_bottleneck = np.where(time_matrix[:, machine] > 0)[0]
        for job in jobs_on_bottleneck:
            bottleneck_influence[job] += 1
    
    # Create composite score efficiently using vectorized operations
    composite_score = (
        total_times * 0.4 +
        time_variances * 0.3 +
        conflict_scores * 0.2 +
        bottleneck_influence * 0.1
    )
    
    # Strategy 4: Strategic time modification for perturbation
    # Apply adaptive perturbations based on job importance and machine constraints
    # Increase processing times for high-impact jobs by 5-15% to create new search space
    perturbation_factor = np.ones((n, m))
    
    # For top 30% of jobs by composite score, apply perturbation
    top_job_indices = np.argsort(-composite_score)[:max(3, n//3)]
    
    for job_idx in top_job_indices:
        # Apply varying perturbations based on job characteristics
        if composite_score[job_idx] > np.percentile(composite_score, 70):
            # High-impact jobs get more aggressive perturbation
            perturbation_factor[job_idx] *= np.random.uniform(1.05, 1.15)
        elif composite_score[job_idx] > np.percentile(composite_score, 30):
            # Medium impact jobs get moderate perturbation
            perturbation_factor[job_idx] *= np.random.uniform(1.02, 1.08)
        else:
            # Low impact jobs get minimal perturbation
            perturbation_factor[job_idx] *= np.random.uniform(1.00, 1.03)
    
    # Apply the perturbations to create new matrix
    new_matrix = time_matrix * perturbation_factor
    
    # Strategy 5: Enhanced job selection for perturbation
    # Select jobs that maximize both impact and diversity
    sorted_indices = np.argsort(-composite_score)
    
    # Select top 5 jobs but ensure diversity in machine usage
    selected_jobs = []
    used_machines = set()
    
    # First pass: select diverse jobs
    for idx in sorted_indices:
        if len(selected_jobs) >= 5:
            break
            
        # Get machines used by this job
        job_machines = set(np.where(time_matrix[idx] > 0)[0])
        
        # If this job doesn't share machines with previously selected jobs, include it
        if not job_machines.intersection(used_machines) or not selected_jobs:
            selected_jobs.append(idx)
            used_machines.update(job_machines)
    
    # Second pass: fill remaining slots with highest scoring jobs
    if len(selected_jobs) < 5:
        remaining_candidates = set(sorted_indices) - set(selected_jobs)
        for idx in sorted_indices:
            if len(selected_jobs) >= 5:
                break
            if idx in remaining_candidates:
                selected_jobs.append(idx)
    
    # Ensure we have at least 2 jobs
    if len(selected_jobs) < 2:
        selected_jobs = list(sorted_indices[:5])
    
    # Limit to exactly 5 jobs maximum
    selected_jobs = selected_jobs[:5]
    
    # Ensure we have at least 2 jobs
    if len(selected_jobs) < 2:
        # Fallback: select first two jobs from sorted list
        selected_jobs = [sorted_indices[0], sorted_indices[1]]
    
    return new_matrix, selected_jobs
# EVOLVE-BLOCK-END
