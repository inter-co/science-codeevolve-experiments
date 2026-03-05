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
    
    # Calculate machine utilization (total processing time per machine)
    machine_utilization = np.sum(time_matrix, axis=0)
    
    # Identify critical machines (those with above-average utilization)
    avg_utilization = np.mean(machine_utilization)
    critical_machines = np.where(machine_utilization > avg_utilization)[0]
    
    # Create penalty matrix based on machine criticality and job-machine interactions
    penalty_matrix = np.zeros_like(time_matrix)
    
    # Apply penalties based on machine criticality
    for machine_idx in critical_machines:
        # Increase processing times for jobs that use critical machines
        # More penalty for jobs with longer processing times on critical machines
        machine_times = time_matrix[:, machine_idx]
        max_time_on_machine = np.max(machine_times)
        
        if max_time_on_machine > 0:
            # Normalize and apply penalty
            normalized_times = machine_times / max_time_on_machine
            penalty_factor = 0.15 * (machine_utilization[machine_idx] / avg_utilization)
            penalty_matrix[:, machine_idx] = penalty_factor * normalized_times
    
    # Add some randomness to prevent over-penalization
    noise = np.random.uniform(0.05, 0.1, size=time_matrix.shape)
    penalty_matrix += noise * (penalty_matrix > 0)
    
    # Apply penalties to the new matrix
    new_matrix = time_matrix * (1.0 + penalty_matrix)
    
    # Enhanced job selection strategy based on multiple criteria:
    # 1. Job-machine conflict severity (jobs that cause bottlenecks)
    # 2. Processing time variance across machines
    # 3. Position in current sequence (favor early jobs for diversity)
    
    # Calculate job-machine conflict scores
    conflict_scores = np.zeros(n)
    for job_idx in range(n):
        job_times = time_matrix[job_idx]
        # Score based on how much this job contributes to critical machine loads
        machine_contribution = np.sum(job_times[critical_machines])
        conflict_scores[job_idx] = machine_contribution
    
    # Calculate processing time variance for each job
    time_variance = np.var(time_matrix, axis=1)
    
    # Combine criteria with weights
    # Higher weight to conflict scores (more critical jobs), 
    # medium to variance (more diverse processing), 
    # lower to position (early jobs preferred)
    position_factor = np.arange(n, 0, -1) / n  # Reverse ranking (higher for earlier jobs)
    
    # Create composite score
    composite_score = (0.5 * conflict_scores / np.max(conflict_scores) + 
                      0.3 * time_variance / np.max(time_variance) + 
                      0.2 * position_factor)
    
    # Select top jobs with highest composite scores
    perturb_jobs = list(np.argsort(-composite_score)[:5])
    
    # Ensure we have at least 2 jobs and no more than 5
    if len(perturb_jobs) < 2:
        # If we don't have enough, add some additional jobs based on total processing time
        total_times = np.sum(time_matrix, axis=1)
        additional_jobs = list(np.argsort(-total_times)[:5-len(perturb_jobs)])
        perturb_jobs.extend([j for j in additional_jobs if j not in perturb_jobs][:5-len(perturb_jobs)])
    
    # Keep only first 5 jobs if needed
    perturb_jobs = perturb_jobs[:5]
    
    # Ensure at least 2 jobs are selected
    if len(perturb_jobs) < 2:
        # Fall back to selecting jobs with highest total processing times
        total_times = np.sum(time_matrix, axis=1)
        perturb_jobs = list(np.argsort(-total_times)[:max(2, min(5, len(perturb_jobs)))])
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
