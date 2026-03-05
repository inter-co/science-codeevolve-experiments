# EVOLVE-BLOCK-START

import numpy as np
from collections import defaultdict
import random

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation builds upon the best practices from inspiration programs
    but introduces a more robust and principled approach:
    1. Combines graph-theoretic conflict analysis with temporal scheduling metrics
    2. Uses a multi-criteria scoring system that balances sensitivity and impact
    3. Applies structured perturbations that create meaningful schedule variations
    4. Guarantees diverse and effective job selection for perturbation

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
    
    # Step 1: Analyze job-machine conflicts and compute centrality measures
    # Create conflict graph based on shared machines
    conflict_graph = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            shared_machines = sum(1 for machine in range(m) 
                                if time_matrix[i, machine] > 0 and time_matrix[j, machine] > 0)
            if shared_machines > 0:
                conflict_graph[i, j] = shared_machines
                conflict_graph[j, i] = shared_machines
    
    # Degree centrality (number of conflicting jobs)
    degree_centrality = np.sum(conflict_graph, axis=1)
    
    # Step 2: Compute machine load statistics
    machine_loads = np.sum(time_matrix, axis=0)
    avg_load = np.mean(machine_loads)
    std_load = np.std(machine_loads)
    
    # Identify overloaded machines (> 1.3 * average load)
    overloaded_machines = np.where(machine_loads > 1.3 * avg_load)[0]
    
    # Step 3: Compute job importance scores based on multiple criteria
    processing_times = np.sum(time_matrix, axis=1)
    processing_variance = np.var(time_matrix, axis=1)
    
    # Normalize all metrics
    norm_proc_times = processing_times / (np.max(processing_times) + 1e-8)
    norm_proc_variance = processing_variance / (np.max(processing_variance) + 1e-8)
    norm_degree = degree_centrality / (np.max(degree_centrality) + 1e-8)
    
    # Compute bottleneck exposure for each job
    bottleneck_exposure = np.zeros(n)
    for i in range(n):
        for machine in overloaded_machines:
            if time_matrix[i, machine] > 0:
                bottleneck_exposure[i] += time_matrix[i, machine]
    norm_bottleneck = bottleneck_exposure / (np.max(processing_times) + 1e-8)
    
    # Compute job importance score with carefully tuned weights
    importance_scores = (
        0.35 * norm_proc_times +           # Processing time importance
        0.25 * norm_proc_variance +        # Processing time variance (sensitivity)
        0.25 * norm_degree +               # Conflict involvement
        0.15 * norm_bottleneck             # Bottleneck exposure
    )
    
    # Step 4: Apply structured perturbations
    # Use a more conservative but effective perturbation approach
    base_perturbation = 0.12  # Base perturbation level
    
    # Apply perturbations based on importance scores
    for job_idx in range(n):
        # Perturbation strength increases with importance
        perturbation_multiplier = 1.0 + base_perturbation * importance_scores[job_idx]
        
        # Apply perturbation to all machines for this job
        for machine_idx in range(m):
            if time_matrix[job_idx, machine_idx] > 0:
                # Add multiplicative noise that preserves relative relationships
                noise_factor = 1.0 + (random.random() - 0.5) * 0.2 * perturbation_multiplier
                new_matrix[job_idx, machine_idx] *= noise_factor
                # Ensure positive values
                new_matrix[job_idx, machine_idx] = max(1, new_matrix[job_idx, machine_idx])
    
    # Step 5: Select jobs for perturbation with guaranteed diversity and quality
    # Sort jobs by importance scores
    sorted_indices = np.argsort(-importance_scores)
    
    # Select top jobs ensuring we get diverse coverage
    selected_jobs = []
    
    # Start with highest importance job
    if len(sorted_indices) > 0:
        selected_jobs.append(current_sequence[sorted_indices[0]])
    
    # Add additional jobs ensuring machine diversity
    machine_coverage = set()
    for job_idx in sorted_indices[1:min(5, len(sorted_indices))]:
        job_id = current_sequence[job_idx]
        # Get machines this job uses
        job_machines = set()
        for machine in range(m):
            if time_matrix[job_id, machine] > 0:
                job_machines.add(machine)
        
        # Only add if this job covers new machines or we haven't filled our quota
        if len(selected_jobs) < 5:
            # Check if this job brings new machine coverage
            new_coverage = job_machines - machine_coverage
            if new_coverage or len(selected_jobs) < 3:  # Always take at least 3 jobs
                selected_jobs.append(job_id)
                machine_coverage.update(job_machines)
    
    # Ensure we have at least 2 jobs
    if len(selected_jobs) < 2:
        # Fill with high-scoring jobs not yet selected
        for job_idx in sorted_indices[1:]:
            job_id = current_sequence[job_idx]
            if job_id not in selected_jobs and len(selected_jobs) < 5:
                selected_jobs.append(job_id)
    
    # Limit to exactly 5 jobs maximum
    selected_jobs = selected_jobs[:5]
    
    # Final safety check to ensure at least 2 jobs
    if len(selected_jobs) < 2:
        # Add remaining jobs from the sequence to meet minimum requirement
        remaining_jobs = [current_sequence[i] for i in range(n) if current_sequence[i] not in selected_jobs]
        while len(selected_jobs) < 2 and remaining_jobs:
            selected_jobs.append(remaining_jobs.pop())
    
    return new_matrix, selected_jobs
# EVOLVE-BLOCK-END
