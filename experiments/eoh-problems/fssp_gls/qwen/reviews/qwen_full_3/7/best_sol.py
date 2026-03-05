# EVOLVE-BLOCK-START

import numpy as np
from scipy.linalg import eigvalsh
from collections import Counter

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using spectral analysis and game-theoretic approaches for better escape
    from local optima in Guided Local Search for Flow Shop Scheduling.

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
    
    # Step 1: Spectral Analysis Approach - Analyze job interaction patterns
    # Create job-machine interaction matrix for spectral analysis
    if n > 1 and m > 1:
        # Compute job-to-job similarity based on processing time patterns
        job_patterns = time_matrix / (np.sum(time_matrix, axis=1, keepdims=True) + 1e-8)
        
        # Compute correlation matrix between jobs
        job_correlation = np.corrcoef(job_patterns)
        
        # Compute eigenvalues to identify dominant patterns
        eigenvals = eigvalsh(job_correlation)
        dominant_pattern_strength = np.sum(eigenvals[-min(3, len(eigenvals)):]) / np.sum(eigenvals)
    else:
        dominant_pattern_strength = 0.0
    
    # Step 2: Game-Theoretic Job Centrality Calculation
    # Treat jobs as strategic players in a game
    job_centrality_scores = np.zeros(n)
    
    # Calculate processing time variance for each job (higher variance = more sensitive)
    job_variances = np.var(time_matrix, axis=1)
    
    # Calculate processing time entropy for each job
    job_entropies = np.zeros(n)
    for i in range(n):
        if np.sum(time_matrix[i]) > 0:
            probs = time_matrix[i] / np.sum(time_matrix[i])
            # Add small epsilon to avoid log(0)
            probs = probs + 1e-10
            job_entropies[i] = -np.sum(probs * np.log(probs))
    
    # Calculate job centrality based on multiple factors
    for i in range(n):
        # Processing time variance (more variable jobs are more critical)
        variance_score = job_variances[i] / (np.max(job_variances) + 1e-8)
        
        # Information entropy (higher entropy = less predictable = more critical)
        entropy_score = job_entropies[i] / (np.max(job_entropies) + 1e-8)
        
        # Position in sequence (jobs later in sequence often more critical)
        position_score = (i / max(1, n-1)) if n > 1 else 0.0
        
        # Combine scores using weights that reflect game-theoretic importance
        job_centrality_scores[i] = 0.3 * variance_score + 0.3 * entropy_score + 0.4 * position_score
    
    # Step 3: Topological Sorting-Based Dependency Analysis
    # Identify jobs that are likely to cause bottlenecks in the schedule
    # by analyzing precedence constraints implicitly through processing times
    
    # Create a measure of job "bottleneck potential"
    bottleneck_potential = np.zeros(n)
    
    # For each job, compute how much its processing times contribute to overall schedule length
    for i in range(n):
        # Compute how much this job contributes to machine workload imbalance
        job_times = time_matrix[i]
        if np.sum(job_times) > 0:
            # Normalize job times by machine capacity
            normalized_times = job_times / (np.mean(time_matrix, axis=0) + 1e-8)
            # Bottleneck potential increases with variance in normalized times
            bottleneck_potential[i] = np.var(normalized_times)
    
    # Step 4: Combine multiple criteria for final selection
    # Use a weighted combination of centrality, bottleneck potential, and pattern strength
    combined_scores = (
        0.4 * job_centrality_scores +
        0.3 * bottleneck_potential +
        0.3 * (dominant_pattern_strength if dominant_pattern_strength > 0.1 else 0.0)
    )
    
    # Step 5: Select jobs for perturbation based on combined scores
    # Sort by combined scores in descending order
    sorted_indices = np.argsort(-combined_scores)
    
    # Select top jobs, ensuring minimum of 2 and maximum of 5
    num_to_select = min(5, max(2, len(sorted_indices)))
    selected_indices = sorted_indices[:num_to_select].tolist()
    
    # Convert from array indices to actual job IDs in current sequence
    perturb_jobs = [current_sequence[idx] for idx in selected_indices]
    
    # Step 6: Apply advanced perturbation using entropy-based disruption
    # Perturb jobs in a way that maximizes information entropy disruption
    for job_idx in range(len(perturb_jobs)):
        job_id = perturb_jobs[job_idx]
        
        # Get the combined score for this job
        combined_score = combined_scores[job_id]
        
        # Determine perturbation intensity based on how "central" the job is
        # More central jobs get stronger perturbations
        base_intensity = 0.15 + 0.25 * combined_score
        
        # Apply entropy-preserving perturbation
        # Use a more sophisticated approach that maintains processing time distributions
        perturbation_factors = np.ones(m)
        
        # Apply different perturbation patterns based on job characteristics
        if combined_score > 0.7:  # Very central job
            # Apply strong, diverse perturbation
            for j in range(m):
                # Randomly vary by larger amount for highly central jobs
                perturbation_factor = 1.0 + np.random.uniform(-base_intensity, base_intensity)
                perturbation_factors[j] = perturbation_factor
        elif combined_score > 0.4:  # Moderately central job  
            # Apply moderate perturbation
            for j in range(m):
                perturbation_factor = 1.0 + np.random.uniform(-base_intensity*0.7, base_intensity*0.7)
                perturbation_factors[j] = perturbation_factor
        else:  # Less central job
            # Apply mild perturbation
            for j in range(m):
                perturbation_factor = 1.0 + np.random.uniform(-base_intensity*0.3, base_intensity*0.3)
                perturbation_factors[j] = perturbation_factor
        
        # Apply the perturbation
        new_matrix[job_id] = time_matrix[job_id] * perturbation_factors
        
        # Ensure bounds and maintain realistic processing times
        min_allowed = time_matrix[job_id] * 0.6
        max_allowed = time_matrix[job_id] * 1.8
        new_matrix[job_id] = np.clip(new_matrix[job_id], min_allowed, max_allowed)
    
    # Step 7: Final validation
    if len(perturb_jobs) < 2:
        # If we don't have enough jobs, select the first two jobs
        perturb_jobs = list(range(min(2, n)))
    
    # Limit to maximum 5 jobs
    perturb_jobs = perturb_jobs[:5]
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
