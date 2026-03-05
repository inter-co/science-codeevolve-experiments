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
    
    # Step 1: Compute completion times for the current sequence (forward pass)
    completion_times = np.zeros((n, m))
    
    # Forward pass to compute completion times
    for job_idx in range(n):
        job_id = current_sequence[job_idx]
        for machine in range(m):
            if machine == 0:
                if job_idx == 0:
                    completion_times[job_idx, machine] = time_matrix[job_id, machine]
                else:
                    completion_times[job_idx, machine] = completion_times[job_idx - 1, machine] + time_matrix[job_id, machine]
            else:
                prev_machine_completion = completion_times[job_idx, machine - 1]
                prev_job_completion = completion_times[job_idx - 1, machine] if job_idx > 0 else 0
                completion_times[job_idx, machine] = max(prev_machine_completion, prev_job_completion) + time_matrix[job_id, machine]
    
    # Step 2: Identify critical jobs and bottlenecks
    # Critical jobs: those that contribute to makespan
    makespan = completion_times[-1, -1]
    
    # Identify jobs that run on heavily loaded machines
    machine_loads = np.sum(time_matrix, axis=0)
    avg_load = np.mean(machine_loads)
    heavy_loaded_machines = np.where(machine_loads > avg_load * 1.2)[0]
    
    # Identify critical jobs based on machine load and position
    critical_jobs = set()
    
    # Jobs that run on heavily loaded machines
    for job_id in range(n):
        for machine in heavy_loaded_machines:
            if time_matrix[job_id, machine] > 0:
                critical_jobs.add(job_id)
    
    # Jobs that are in the last portion of the schedule (potential critical path)
    critical_window_size = min(5, n // 4)
    for job_idx in range(n - critical_window_size, n):
        if job_idx >= 0:
            job_id = current_sequence[job_idx]
            critical_jobs.add(job_id)
    
    # Step 3: Apply strategic penalties to create perturbation opportunities
    base_penalty = 0.2
    
    # Apply penalties based on job characteristics
    for job_id in range(n):
        penalty_factor = 1.0
        
        # Higher penalty for critical jobs
        if job_id in critical_jobs:
            penalty_factor += 0.5
            
        # Apply penalty with some randomness for diversity
        penalty = base_penalty * penalty_factor * (1.0 + np.random.random() * 0.4)
        new_matrix[job_id] = time_matrix[job_id] * (1.0 + penalty)
    
    # Step 4: Select jobs for perturbation using weighted approach
    # Calculate weights for each job based on importance
    job_weights = np.zeros(n)
    
    for job_idx in range(n):
        job_id = current_sequence[job_idx]
        weight = 1.0
        
        # Higher weight for critical jobs
        if job_id in critical_jobs:
            weight += 0.7
            
        # Higher weight for jobs with high processing times
        processing_time = np.sum(time_matrix[job_id])
        max_proc_time = np.max(np.sum(time_matrix, axis=1))
        if max_proc_time > 0:
            weight += 0.3 * (processing_time / max_proc_time)
            
        job_weights[job_idx] = weight
    
    # Normalize weights
    if np.sum(job_weights) > 0:
        job_weights = job_weights / np.sum(job_weights)
    else:
        # If all weights are zero, give equal chance to all jobs
        job_weights = np.ones(n) / n
    
    # Use weighted sampling to select jobs
    num_perturb_jobs = min(5, max(2, int(np.random.random() * 4) + 2))
    
    # Sample jobs ensuring we don't sample duplicates
    selected_indices = np.random.choice(n, size=min(num_perturb_jobs, n), p=job_weights, replace=False)
    
    # Convert back to actual job IDs
    perturb_jobs = [current_sequence[i] for i in selected_indices]
    
    # Ensure we always return at least 2 jobs
    if len(perturb_jobs) < 2:
        perturb_jobs = [current_sequence[0], current_sequence[1]]
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
