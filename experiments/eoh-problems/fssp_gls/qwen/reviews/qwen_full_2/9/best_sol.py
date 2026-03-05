# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using an enhanced approach combining bottleneck detection and strategic sampling.

    This implementation focuses on:
    1. Identifying critical bottleneck machines and jobs
    2. Using variance-based importance measures
    3. Strategic sampling for job selection with diversity
    4. Adaptive perturbation with controlled randomness
    5. Guaranteed minimum coverage for effective escape from local optima

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
    
    # Step 1: Identify machine bottlenecks by analyzing load variance
    machine_loads = np.sum(time_matrix, axis=0)
    avg_load = np.mean(machine_loads)
    std_load = np.std(machine_loads)
    
    # Jobs that are processed on overloaded machines are more critical
    bottleneck_machine_mask = machine_loads > (avg_load + 0.5 * std_load)
    
    # Step 2: Compute comprehensive job importance scores
    job_importance = np.zeros(n)
    
    # Criterion 1: Processing time variance across machines (higher variance = more sensitive)
    job_variance = np.var(time_matrix, axis=1)
    
    # Criterion 2: Position in sequence (jobs at boundaries have more flexibility)
    positional_score = np.zeros(n)
    for i in range(n):
        pos_ratio = i / (n - 1) if n > 1 else 0.5
        # Jobs at extremes have higher potential for impact (0.5 at center, 1.0 at ends)
        positional_score[i] = 1.0 - abs(pos_ratio - 0.5)
    
    # Criterion 3: Total processing time (longer jobs often more important)
    total_processing_times = np.sum(time_matrix, axis=1)
    
    # Criterion 4: Bottleneck exposure (jobs processed on bottleneck machines)
    bottleneck_exposure = np.zeros(n)
    for i in range(n):
        # Count how many machines this job is processed on that are bottlenecks
        bottleneck_count = sum(1 for j in range(m) if time_matrix[i, j] > 0 and bottleneck_machine_mask[j])
        bottleneck_exposure[i] = bottleneck_count / max(1, m)
    
    # Combine criteria with optimized weights
    job_importance = (
        0.35 * job_variance + 
        0.25 * positional_score + 
        0.30 * total_processing_times +
        0.10 * bottleneck_exposure
    )
    
    # Step 3: Apply adaptive perturbations
    max_importance = np.max(job_importance) + 1e-8
    
    for i in range(n):
        # Normalize importance score
        normalized_importance = job_importance[i] / max_importance
        
        # Apply perturbation - more important jobs get stronger perturbations
        # Base perturbation: 5% to 35% change (more aggressive than before)
        base_perturbation = 0.05 + 0.30 * normalized_importance
        
        # Add controlled randomness with wider range for more exploration
        noise = np.random.uniform(-0.15, 0.15)
        perturbation_factor = 1.0 + base_perturbation + noise
        
        # Clip to reasonable bounds to prevent extreme changes
        perturbation_factor = np.clip(perturbation_factor, 0.7, 1.4)
        
        # Apply to all machines for this job
        new_matrix[i] = time_matrix[i] * perturbation_factor
    
    # Step 4: Strategic job selection with guaranteed diversity
    # Sort by importance descending
    sorted_indices = np.argsort(-job_importance)
    
    # Select 3-5 jobs with highest importance scores
    # Use dynamic selection based on problem size for better scaling
    if n <= 10:
        num_selected = min(3, n)
    elif n <= 20:
        num_selected = min(4, n)
    else:
        num_selected = min(5, n)
    
    # Always include the top 2 jobs for reliability
    perturb_jobs = sorted_indices[:2].tolist()
    
    # Add remaining jobs strategically
    remaining_needed = num_selected - len(perturb_jobs)
    if remaining_needed > 0:
        # Add jobs from the top-ranked list, but ensure diversity
        added_count = 0
        for idx in sorted_indices[2:]:
            if added_count >= remaining_needed:
                break
            if idx not in perturb_jobs:
                perturb_jobs.append(idx)
                added_count += 1
    
    # Ensure we have at least 2 jobs and at most 5
    if len(perturb_jobs) < 2:
        perturb_jobs = sorted_indices[:2].tolist()
    elif len(perturb_jobs) > 5:
        perturb_jobs = perturb_jobs[:5]
    
    # Step 5: Enhance diversity by ensuring spread across sequence
    # If we have a longer sequence and have fewer jobs, add more diverse selections
    if len(current_sequence) >= 5 and len(perturb_jobs) < 5:
        # Add jobs from different segments of the sequence to increase diversity
        segment_size = max(1, len(current_sequence) // 3)
        for i in range(3):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, len(current_sequence))
            if start_idx < end_idx:
                # Pick one job from this segment that's not already selected
                segment_jobs = current_sequence[start_idx:end_idx]
                for job in segment_jobs:
                    if job not in perturb_jobs and len(perturb_jobs) < 5:
                        perturb_jobs.append(job)
                        break
    
    # Final validation - ensure we have at least 2 jobs
    if len(perturb_jobs) < 2:
        # Fall back to first two jobs if needed
        perturb_jobs = list(range(min(2, n)))
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
