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
    
    # Step 1: Identify bottleneck machines based on total workload
    machine_loads = np.sum(time_matrix, axis=0)
    avg_load = np.mean(machine_loads)
    
    # Bottleneck machines are those with load significantly above average
    bottleneck_machines = np.where(machine_loads > avg_load * 1.2)[0]
    
    # Step 2: Calculate job criticality scores
    job_criticalities = np.zeros(n)
    
    # Normalize processing times for fair comparison
    total_processing_times = np.sum(time_matrix, axis=1)
    max_proc_time = np.max(total_processing_times)
    
    # Calculate position-based criticality (jobs later in sequence are more critical)
    if n > 0:
        position_criticality = np.arange(n) / n
    else:
        position_criticality = np.zeros(n)
    
    # Vectorized calculation of bottleneck contributions
    if len(bottleneck_machines) > 0 and n > 0:
        # Create a mask for bottleneck machines
        bottleneck_mask = np.zeros(m, dtype=bool)
        bottleneck_mask[bottleneck_machines] = True
        
        # For each job, calculate contribution to bottleneck loads
        bottleneck_contributions = np.zeros(n)
        for i in range(n):
            contrib = 0
            for j in range(m):
                if bottleneck_mask[j] and time_matrix[i, j] > 0:
                    contrib += time_matrix[i, j] / (machine_loads[j] if machine_loads[j] > 0 else 1)
            bottleneck_contributions[i] = contrib
    else:
        bottleneck_contributions = np.zeros(n)
    
    # Combine all factors into criticality scores
    for i in range(n):
        # Normalized processing time criticality
        proc_criticality = total_processing_times[i] / max_proc_time if max_proc_time > 0 else 0
        
        # Position-based criticality (later jobs more important)
        pos_criticality = position_criticality[i]
        
        # Bottleneck contribution
        bottleneck_criticality = bottleneck_contributions[i]
        
        # Weighted combination
        job_criticalities[i] = 0.4 * proc_criticality + 0.3 * pos_criticality + 0.3 * bottleneck_criticality
    
    # Step 3: Select jobs for perturbation
    # Sort by criticality and take top jobs
    sorted_indices = np.argsort(-job_criticalities)
    
    # Select 2-5 jobs based on criticality scores
    # Ensure we have at least 2 jobs, but not more than 5
    num_to_select = min(5, max(2, len(sorted_indices)))
    selected_indices = sorted_indices[:num_to_select].tolist()
    
    # Convert from array indices to actual job IDs in current sequence
    perturb_jobs = [current_sequence[idx] for idx in selected_indices]
    
    # Step 4: Apply structured perturbation to selected jobs
    # Use multiplicative perturbation to maintain realistic processing times
    for job_id in perturb_jobs:
        # Find the index of this job in the current sequence
        job_idx = current_sequence.index(job_id)
        
        # Get criticality score for this job
        crit_score = job_criticalities[job_idx]
        
        # Determine perturbation intensity based on criticality
        # More critical jobs get stronger perturbations
        base_perturbation = 0.1 + 0.3 * crit_score
        
        # Apply multiplicative perturbation to each machine
        perturbation = np.ones(m)
        for i in range(m):
            # Apply perturbation in a controlled way
            perturbation[i] = 1.0 + np.random.uniform(-base_perturbation, base_perturbation)
        
        # Apply perturbation
        new_matrix[job_id] = time_matrix[job_id] * perturbation
        
        # Ensure all processing times remain reasonable (not too small)
        new_matrix[job_id] = np.maximum(new_matrix[job_id], time_matrix[job_id] * 0.7)
    
    # Step 5: Final validation
    # Ensure we have at least 2 jobs to perturb
    if len(perturb_jobs) < 2:
        # If we don't have enough jobs, select the first two jobs
        perturb_jobs = list(range(min(2, n)))
    
    # Limit to maximum 5 jobs
    perturb_jobs = perturb_jobs[:5]
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
