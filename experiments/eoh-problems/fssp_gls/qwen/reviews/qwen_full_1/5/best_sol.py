# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation uses a novel clustering-based approach that:
    1. Identifies diverse job groups based on processing time patterns
    2. Applies adaptive perturbation to machine bottlenecks
    3. Uses statistical sampling for exploration diversity
    4. Maintains computational efficiency while improving perturbation quality

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
    
    # Strategy 1: Statistical clustering approach for job grouping
    # Cluster jobs based on their processing time distributions across machines
    job_means = np.mean(time_matrix, axis=1)
    job_stds = np.std(time_matrix, axis=1)
    
    # Normalize and create feature vectors for clustering
    normalized_means = job_means / (np.max(job_means) + 1e-8)
    normalized_stds = job_stds / (np.max(job_stds) + 1e-8)
    
    # Create feature matrix for clustering
    features = np.column_stack([normalized_means, normalized_stds])
    
    # Simple k-means style clustering (k=3) - this is much faster than full clustering
    # We'll cluster based on the mean processing time variation
    if n >= 3:
        # Determine cluster boundaries based on quantiles
        q1 = np.percentile(job_means, 33)
        q2 = np.percentile(job_means, 66)
        
        # Assign jobs to clusters
        cluster_assignments = np.zeros(n, dtype=int)
        cluster_assignments[job_means <= q1] = 0  # Low processing time group
        cluster_assignments[(job_means > q1) & (job_means <= q2)] = 1  # Medium group
        cluster_assignments[job_means > q2] = 2  # High processing time group
    else:
        # For small instances, assign all to same cluster or simple grouping
        cluster_assignments = np.zeros(n, dtype=int)
    
    # Strategy 2: Machine load analysis and bottleneck identification
    machine_loads = np.sum(time_matrix, axis=0)
    load_deviation = np.abs(machine_loads - np.mean(machine_loads))
    bottleneck_machines = np.where(load_deviation > np.std(machine_loads))[0]
    
    # Strategy 3: Adaptive perturbation matrix modification
    # Increase processing times for jobs in high-load scenarios
    # This creates more challenging scheduling conditions
    
    # Base penalty factor
    base_penalty = 0.15
    
    # Apply different penalties based on job characteristics
    penalty_factors = np.ones(n)
    
    # Give higher penalties to jobs in bottleneck situations
    for i in range(n):
        # Check if job contributes significantly to any bottleneck machine
        job_machine_loads = time_matrix[i]
        high_load_machine_count = sum(1 for j in bottleneck_machines 
                                    if job_machine_loads[j] > np.mean(job_machine_loads) * 1.2)
        
        # Jobs that contribute to multiple bottlenecks get higher penalties
        penalty_factors[i] += base_penalty * min(high_load_machine_count, 3) * 0.5
    
    # Apply penalties to the time matrix with some randomness
    # This introduces stochasticity to avoid getting stuck in similar patterns
    random_factor = np.random.uniform(0.8, 1.2, n)
    adjusted_penalty_factors = penalty_factors * random_factor
    
    for i in range(n):
        new_matrix[i] = time_matrix[i] * (1.0 + adjusted_penalty_factors[i] * 0.5)
    
    # Strategy 4: Job selection for perturbation using diversity sampling
    # Select jobs from different clusters and load contexts for maximum exploration
    
    # Start by selecting jobs from each cluster
    selected_clusters = set()
    selected_indices = []
    
    # Sample from each cluster ensuring diversity
    for cluster_id in range(3):
        cluster_mask = cluster_assignments == cluster_id
        cluster_jobs = np.where(cluster_mask)[0]
        
        if len(cluster_jobs) > 0:
            # Select one representative job from this cluster
            # Prefer jobs with higher variability (std) for more impact
            if len(cluster_jobs) == 1:
                representative = cluster_jobs[0]
            else:
                # Select job with highest standard deviation in this cluster
                representative = cluster_jobs[np.argmax(job_stds[cluster_jobs])]
            
            selected_indices.append(representative)
            selected_clusters.add(cluster_id)
            
            if len(selected_indices) >= 3:
                break
    
    # Strategy 5: Add jobs based on machine load context
    # Include jobs that are processed on bottleneck machines
    if len(selected_indices) < 5 and len(bottleneck_machines) > 0:
        # Find jobs that have high processing times on bottleneck machines
        for i in range(n):
            job_bottleneck_load = sum(time_matrix[i][j] for j in bottleneck_machines)
            if job_bottleneck_load > np.mean(machine_loads[bottleneck_machines]) * 0.8:
                if i not in selected_indices:
                    selected_indices.append(i)
                    if len(selected_indices) >= 5:
                        break
    
    # Strategy 6: Ensure minimum diversity and completeness
    # If we don't have enough diverse jobs, fill with high-impact jobs
    if len(selected_indices) < 2:
        # Fall back to highest total processing time jobs
        high_time_indices = np.argsort(-np.sum(time_matrix, axis=1))[:5]
        selected_indices = high_time_indices[:2].tolist()
    
    # Fill remaining slots with jobs that have high variability or are from different clusters
    if len(selected_indices) < 5:
        # Add jobs that haven't been selected yet, prioritizing high variance
        available_indices = set(range(n)) - set(selected_indices)
        if available_indices:
            # Sort remaining jobs by processing time variance
            remaining_indices = list(available_indices)
            remaining_variances = [job_stds[i] for i in remaining_indices]
            sorted_remaining = [i for _, i in sorted(zip(remaining_variances, remaining_indices), reverse=True)]
            
            for i in sorted_remaining:
                if len(selected_indices) >= 5:
                    break
                if i not in selected_indices:
                    selected_indices.append(i)
    
    # Limit to exactly 5 jobs (or fewer if needed) and remove duplicates
    selected_indices = list(dict.fromkeys(selected_indices))[:5]
    
    # Convert to actual job indices in the current sequence
    perturb_jobs = [current_sequence[i] for i in selected_indices[:5]]
    
    # Final validation to ensure at least 2 jobs
    if len(perturb_jobs) < 2:
        # Force inclusion of two highest total time jobs
        high_time_indices = np.argsort(-np.sum(time_matrix, axis=1))[:2]
        perturb_jobs = [current_sequence[i] for i in high_time_indices]
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
