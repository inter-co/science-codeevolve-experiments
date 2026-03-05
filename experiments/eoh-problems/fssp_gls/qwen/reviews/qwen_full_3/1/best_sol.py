# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation uses a direct critical path analysis approach:
    1. Identifies jobs that are most critical to the makespan
    2. Applies perturbations based on job criticality
    3. Selects diverse jobs for perturbation based on multiple criteria

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
    
    # Step 1: Critical path analysis using forward and backward pass
    # Forward pass to compute earliest start times
    forward_times = np.zeros((n, m))
    forward_times[:, 0] = time_matrix[:, 0]
    for j in range(1, m):
        forward_times[:, j] = forward_times[:, j-1] + time_matrix[:, j]
    
    # Backward pass to compute latest finish times  
    backward_times = np.zeros((n, m))
    backward_times[:, m-1] = forward_times[:, m-1]
    for j in range(m-2, -1, -1):
        backward_times[:, j] = backward_times[:, j+1] - time_matrix[:, j]
    
    # Critical path analysis: jobs with zero slack are critical
    slack = backward_times - forward_times
    critical_slack = np.min(slack, axis=1)  # Minimum slack per job
    
    # Step 2: Machine bottleneck analysis
    # Identify machines with high utilization
    machine_loads = np.sum(time_matrix, axis=0)
    avg_load = np.mean(machine_loads)
    machine_utilization = machine_loads / (avg_load + 1e-8)
    
    # Find machines with utilization > 1.2 (overloaded)
    overloaded_machines = np.where(machine_utilization > 1.2)[0]
    
    # Step 3: Job importance scoring based on criticality and machine impact
    job_scores = np.zeros(n)
    
    # Criterion 1: Critical path position (lower slack = more critical)
    # Normalize slack scores so that lowest slack gets highest score
    normalized_slack = -critical_slack  # Negate to make lower slack higher score
    max_slack = np.max(normalized_slack)
    if max_slack > 0:
        job_scores += (normalized_slack / max_slack) * 0.4
    
    # Criterion 2: Machine overload impact
    # Jobs that appear on overloaded machines get higher scores
    machine_impact = np.zeros(n)
    for i in range(n):
        job_machines = np.where(time_matrix[i] > 0)[0]
        impact_score = sum(1 for machine in job_machines if machine in overloaded_machines)
        machine_impact[i] = impact_score
    
    max_machine_impact = np.max(machine_impact)
    if max_machine_impact > 0:
        job_scores += (machine_impact / max_machine_impact) * 0.3
    
    # Criterion 3: Total processing time (larger jobs are often more impactful)
    total_processing = np.sum(time_matrix, axis=1)
    max_processing = np.max(total_processing)
    if max_processing > 0:
        job_scores += (total_processing / max_processing) * 0.2
    
    # Criterion 4: Positional importance (jobs near boundaries are more sensitive)
    boundary_importance = np.zeros(n)
    if len(current_sequence) > 0:
        for pos, job_id in enumerate(current_sequence):
            # Distance to nearest boundary
            dist_to_boundary = min(pos, len(current_sequence) - 1 - pos)
            # Convert to importance score (0 = center, 1 = boundary)
            importance = 1.0 - (dist_to_boundary / max(1, len(current_sequence) // 2))
            boundary_importance[job_id] = importance
    
    job_scores += boundary_importance * 0.1
    
    # Step 4: Apply perturbations based on job importance
    # Use a more direct approach: perturb jobs with highest scores
    # But also ensure some diversity in the perturbation
    
    # Identify top jobs by score
    top_jobs = np.argsort(-job_scores)[:min(5, n)]
    
    # Apply perturbations with varying intensities based on importance
    # Most important jobs get stronger perturbations
    for i, job_idx in enumerate(top_jobs):
        # Stronger perturbation for top 2 jobs, moderate for next 2, light for last
        if i < 2:
            # Strong perturbation (±15%)
            perturbation_factor = 1.0 + (np.random.random() - 0.5) * 0.3
        elif i < 4:
            # Moderate perturbation (±8%)
            perturbation_factor = 1.0 + (np.random.random() - 0.5) * 0.16
        else:
            # Light perturbation (±3%)
            perturbation_factor = 1.0 + (np.random.random() - 0.5) * 0.06
            
        new_matrix[job_idx] = time_matrix[job_idx] * perturbation_factor
    
    # Step 5: Enhanced job selection for perturbation
    # Ensure we have enough diverse jobs to perturb
    # Start with the highest scoring jobs
    selected_jobs = list(top_jobs)
    
    # Add additional jobs to ensure we have at least 2 and up to 5
    if len(selected_jobs) < 2:
        # Fill with jobs that have high processing times
        high_proc_jobs = np.argsort(-total_processing)[:5]
        for job_idx in high_proc_jobs:
            if len(selected_jobs) >= 5:
                break
            if job_idx not in selected_jobs:
                selected_jobs.append(job_idx)
    elif len(selected_jobs) < 5:
        # Add some jobs with high machine impact for diversity
        high_impact_jobs = np.argsort(-machine_impact)[:5]
        for job_idx in high_impact_jobs:
            if len(selected_jobs) >= 5:
                break
            if job_idx not in selected_jobs:
                selected_jobs.append(job_idx)
    
    # Limit to 5 jobs maximum
    perturb_jobs = selected_jobs[:5]
    
    # Ensure minimum size of 2
    if len(perturb_jobs) < 2:
        # Add jobs with high total processing time
        remaining_jobs = [i for i in range(n) if i not in perturb_jobs]
        needed = 2 - len(perturb_jobs)
        perturb_jobs.extend(remaining_jobs[:needed])
    
    # Sort for deterministic behavior
    perturb_jobs.sort()
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
