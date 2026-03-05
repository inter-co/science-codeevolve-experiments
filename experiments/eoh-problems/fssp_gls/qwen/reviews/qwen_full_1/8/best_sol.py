# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.spatial.distance import pdist, squareform

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using a novel information-theoretic and network-flow based approach for GLS.
    
    This implementation leverages:
    1. Information-theoretic measures for schedule uncertainty quantification
    2. Network flow analysis for job-machine dependency modeling
    3. Entropy-based job prioritization for exploration
    4. Graph cut methods for diverse job selection

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
    
    # INFORMATION-THEORETIC APPROACH: Calculate entropy of processing patterns
    # High entropy indicates more diverse/uncertain processing patterns
    job_entropies = np.zeros(n)
    machine_entropies = np.zeros(m)
    
    # Calculate job entropy based on processing time distribution across machines
    for i in range(n):
        job_times = time_matrix[i]
        # Normalize to probability distribution
        if np.sum(job_times) > 0:
            prob_dist = job_times / np.sum(job_times)
            # Avoid log(0) by adding small epsilon
            prob_dist = np.clip(prob_dist, 1e-10, 1.0)
            job_entropies[i] = -np.sum(prob_dist * np.log2(prob_dist))
        else:
            job_entropies[i] = 0
    
    # Calculate machine entropy (how evenly jobs are distributed)
    for j in range(m):
        machine_times = time_matrix[:, j]
        if np.sum(machine_times) > 0:
            prob_dist = machine_times / np.sum(machine_times)
            prob_dist = np.clip(prob_dist, 1e-10, 1.0)
            machine_entropies[j] = -np.sum(prob_dist * np.log2(prob_dist))
        else:
            machine_entropies[j] = 0
    
    # NETWORK-FLOW ANALYSIS: Create job-machine bipartite graph
    # Compute edge weights based on processing time ratios
    job_machine_ratios = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if time_matrix[i, j] > 0:
                # Ratio of job time to machine average time
                machine_avg = np.mean(time_matrix[:, j])
                if machine_avg > 0:
                    job_machine_ratios[i, j] = time_matrix[i, j] / machine_avg
                else:
                    job_machine_ratios[i, j] = 1.0
            else:
                job_machine_ratios[i, j] = 0.0
    
    # Calculate job "conflict" scores based on ratio variance
    job_conflicts = np.var(job_machine_ratios, axis=1)
    
    # CRITICAL PATH ANALYSIS: Identify jobs that might be on critical paths
    # Using a simplified forward-backward pass approach
    job_completion_times = np.zeros((n, m))
    
    # Forward pass: calculate earliest completion times
    for i in range(n):
        for j in range(m):
            if j == 0:
                job_completion_times[i, j] = time_matrix[i, j]
            else:
                job_completion_times[i, j] = max(
                    job_completion_times[i, j-1],
                    job_completion_times[current_sequence.index(i), j-1]
                ) + time_matrix[i, j]
    
    # Backward pass: calculate latest completion times
    job_latest_times = np.zeros((n, m))
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            if j == m-1:
                job_latest_times[i, j] = job_completion_times[i, j]
            else:
                job_latest_times[i, j] = min(
                    job_latest_times[i, j+1],
                    job_latest_times[current_sequence.index(i), j+1]
                ) - time_matrix[i, j]
    
    # Critical path score: difference between earliest and latest times
    critical_path_scores = np.zeros(n)
    for i in range(n):
        # For simplicity, use the maximum difference in completion times
        max_diff = np.max(job_completion_times[i] - job_latest_times[i])
        critical_path_scores[i] = max_diff
    
    # ENTROPY-BASED JOB PRIORITIZATION
    # Jobs with high entropy (more uncertain/sensitive) should be perturbed
    # Jobs with high conflict scores (inconsistent patterns) should be perturbed
    # Jobs with high critical path scores (potential bottlenecks) should be perturbed
    
    # Normalize all scores to [0,1] range
    def normalize_scores(scores):
        max_score = np.max(scores)
        if max_score > 0:
            return scores / max_score
        return scores
    
    normalized_job_entropies = normalize_scores(job_entropies)
    normalized_job_conflicts = normalize_scores(job_conflicts)
    normalized_critical_scores = normalize_scores(critical_path_scores)
    
    # Combine scores using adaptive weights based on problem characteristics
    # In this case, we'll use a dynamic weighting scheme
    # Higher entropy = more exploration needed
    # Higher conflicts = more instability in scheduling
    # Higher critical path = more potential for improvement
    
    # Weights that favor exploration and stability
    weights = np.array([0.4, 0.3, 0.3])  # entropy, conflicts, critical path
    
    combined_scores = (
        weights[0] * normalized_job_entropies +
        weights[1] * normalized_job_conflicts +
        weights[2] * normalized_critical_scores
    )
    
    # APPLY PERTURBATION: Use network flow-inspired modification
    # Instead of just increasing times, we modify the structure to create
    # a more challenging search landscape
    
    # Select top 30% of jobs based on combined scores
    top_indices = np.argsort(-combined_scores)[:max(2, n // 3)]
    
    # Apply perturbations using a more sophisticated approach:
    # 1. Increase processing times slightly (to shift the landscape)
    # 2. Add structured noise based on machine correlations
    # 3. Create artificial bottlenecks to force reevaluation
    
    for idx in top_indices:
        # Calculate correlation matrix for this job across machines
        job_times = time_matrix[idx]
        if np.sum(job_times) > 0:
            # Add structured noise that preserves relative relationships
            noise_factor = 0.1  # 10% noise
            
            # Generate correlated noise based on machine relationships
            machine_correlations = np.corrcoef(time_matrix.T)
            # Ensure diagonal is 1 (self-correlation)
            np.fill_diagonal(machine_correlations, 1.0)
            
            # Apply noise to job times
            # Use a random vector to create correlated perturbations
            noise_vector = np.random.normal(0, noise_factor, m)
            
            # Apply structured perturbation
            new_matrix[idx] = np.maximum(
                time_matrix[idx] * (1.0 + noise_vector),
                time_matrix[idx] * 0.8  # Minimum 80% of original
            )
    
    # JOB SELECTION: Use graph-based diversity selection
    # Create distance matrix between jobs based on their processing patterns
    pattern_distances = squareform(pdist(time_matrix, metric='euclidean'))
    
    # Select jobs that maximize diversity in the selection
    # This uses a greedy approach to select diverse jobs
    selected_indices = []
    
    # Always include the highest scoring job
    best_idx = np.argmax(combined_scores)
    selected_indices.append(best_idx)
    
    # Then select additional jobs that maximize diversity
    remaining_indices = set(range(n)) - {best_idx}
    
    # Select up to 4 more jobs that are maximally distant from already selected
    while len(selected_indices) < min(5, n) and remaining_indices:
        # Calculate average distance to already selected jobs
        avg_distances = np.zeros(len(remaining_indices))
        remaining_list = list(remaining_indices)
        
        for i, idx in enumerate(remaining_list):
            distances = [pattern_distances[idx, sel_idx] for sel_idx in selected_indices]
            avg_distances[i] = np.mean(distances) if distances else 0
        
        # Select the job with maximum average distance
        if len(avg_distances) > 0:
            max_dist_idx = np.argmax(avg_distances)
            selected_idx = remaining_list[max_dist_idx]
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
        else:
            break
    
    # Convert to actual job indices in current sequence
    perturb_jobs = [current_sequence[i] for i in selected_indices[:5]]
    
    # Ensure minimum of 2 jobs (fallback mechanism)
    if len(perturb_jobs) < 2:
        # Select based on highest processing times as fallback
        total_processing = np.sum(time_matrix, axis=1)
        sorted_by_proc = np.argsort(-total_processing)
        perturb_jobs = [current_sequence[i] for i in sorted_by_proc[:min(5, len(sorted_by_proc))]]
        perturb_jobs = perturb_jobs[:5]
    
    # Return first 5 jobs (ensured to have at least 2)
    perturb_jobs = perturb_jobs[:5]
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
