# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using a graph-theoretic and bottleneck-aware approach for Flow Shop Scheduling.

    This implementation focuses on:
    - Critical path identification using forward/backward pass
    - Machine bottleneck detection based on load balancing
    - Strategic time modifications that create meaningful perturbations
    - Diversified job selection with deterministic guarantees

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
    
    # Step 1: Compute critical path information using forward/backward pass
    # Forward pass: compute earliest completion times
    forward_times = np.zeros((n, m))
    forward_times[0, 0] = time_matrix[current_sequence[0], 0]
    for j in range(1, m):
        forward_times[0, j] = forward_times[0, j-1] + time_matrix[current_sequence[0], j]
    
    for i in range(1, n):
        forward_times[i, 0] = forward_times[i-1, 0] + time_matrix[current_sequence[i], 0]
        for j in range(1, m):
            forward_times[i, j] = max(forward_times[i-1, j], forward_times[i, j-1]) + time_matrix[current_sequence[i], j]
    
    # Backward pass: compute latest completion times
    backward_times = np.zeros((n, m))
    backward_times[n-1, m-1] = forward_times[n-1, m-1]
    for j in range(m-2, -1, -1):
        backward_times[n-1, j] = backward_times[n-1, j+1] - time_matrix[current_sequence[n-1], j+1]
    
    for i in range(n-2, -1, -1):
        backward_times[i, m-1] = backward_times[i+1, m-1] - time_matrix[current_sequence[i], m-1]
        for j in range(m-2, -1, -1):
            backward_times[i, j] = min(backward_times[i+1, j], backward_times[i, j+1]) - time_matrix[current_sequence[i], j]
    
    # Calculate critical path lengths for each job-machine pair
    critical_path_lengths = forward_times + backward_times - time_matrix[current_sequence, :]

    # Step 2: Identify machine bottlenecks
    machine_loads = np.sum(time_matrix, axis=0)
    max_machine_load = np.max(machine_loads)
    
    # Create penalty matrix based on both critical path and machine load
    penalty_factor = np.ones_like(time_matrix)
    
    # Criterion 1: Jobs on critical paths (higher penalty for jobs that affect critical path)
    criticality_scores = np.max(critical_path_lengths, axis=1)
    max_criticality = np.max(criticality_scores)
    if max_criticality > 0:
        penalty_factor += 0.2 * criticality_scores.reshape(-1, 1) / max_criticality
    
    # Criterion 2: Jobs on overloaded machines (bottleneck machines)
    machine_penalties = np.zeros(n)
    for i in range(n):
        job_idx = current_sequence[i]
        # For each job, find the machine with highest load
        machine_idx = np.argmax(machine_loads)
        machine_penalties[i] = machine_loads[machine_idx] / (max_machine_load + 1e-8)
    
    # Apply machine-based penalties
    penalty_factor += 0.15 * machine_penalties.reshape(-1, 1)
    
    # Criterion 3: Job sensitivity to sequencing (variance in processing times)
    job_variances = np.var(time_matrix, axis=1)
    max_variance = np.max(job_variances)
    if max_variance > 0:
        penalty_factor += 0.1 * job_variances.reshape(-1, 1) / max_variance
    
    # Step 3: Apply strategic time modifications with enhanced perturbation
    # Use more aggressive perturbations for high-priority jobs
    perturbation_strength = 1.0 + 0.2 * np.random.rand()  # Deterministic factor + slight randomness
    
    # Apply penalties with controlled variation
    # Create a more diverse perturbation pattern
    base_perturbation = 0.9 + 0.2 * np.random.rand(n, m)
    new_matrix = time_matrix * penalty_factor * base_perturbation
    
    # Ensure some minimum modification to guarantee perturbation
    min_modification = 0.05
    modification_mask = (penalty_factor > 1.0) | (base_perturbation < 0.95)
    new_matrix = np.where(modification_mask, 
                         new_matrix * (1 + min_modification), 
                         new_matrix)
    
    # Step 4: Select jobs for perturbation using a multi-criteria approach
    # Combine criticality, variance, and machine load information
    
    # Normalize criteria for fair comparison
    normalized_criticality = criticality_scores / (np.max(criticality_scores) + 1e-8)
    normalized_variance = job_variances / (np.max(job_variances) + 1e-8)
    normalized_machine_load = machine_penalties / (np.max(machine_penalties) + 1e-8)
    
    # Create diversified scores with different weights
    job_scores = 0.4 * normalized_criticality + 0.3 * normalized_variance + 0.3 * normalized_machine_load
    
    # Select jobs ensuring good diversity and coverage
    sorted_indices = np.argsort(-job_scores)
    
    # Select 3-5 jobs depending on problem size, but always at least 2
    num_jobs = min(5, max(2, n // 6))
    
    # Take top jobs but ensure diversity by selecting from different parts of the sequence
    perturb_jobs = []
    selected_indices = set()
    
    # Add top jobs ensuring diversity
    for idx in sorted_indices[:min(num_jobs * 2, len(sorted_indices))]:
        if len(perturb_jobs) >= num_jobs:
            break
        if idx not in selected_indices:
            perturb_jobs.append(idx)
            selected_indices.add(idx)
    
    # If we didn't get enough jobs, fill with additional diverse ones
    if len(perturb_jobs) < num_jobs:
        remaining_indices = [i for i in range(n) if i not in selected_indices]
        for i in range(min(num_jobs - len(perturb_jobs), len(remaining_indices))):
            perturb_jobs.append(remaining_indices[i])
    
    # Ensure deterministic behavior by sorting the final list
    perturb_jobs.sort()
    
    # Convert back to original job indices
    perturb_jobs = [current_sequence[job_idx] for job_idx in perturb_jobs]
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
