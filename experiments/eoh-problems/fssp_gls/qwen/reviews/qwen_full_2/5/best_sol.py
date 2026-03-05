# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation uses a memory-aware, stochastic perturbation strategy
    that balances exploration and exploitation effectively.

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
    
    # Step 1: Simple but effective job importance calculation
    # Using processing time statistics to identify critical jobs
    job_processing_times = np.sum(time_matrix, axis=1)
    
    # Variance in processing times (higher variance = more sensitive to changes)
    job_variance = np.var(time_matrix, axis=1)
    
    # Total processing time normalized by machine count (workload indicator)
    job_workload = job_processing_times / m
    
    # Compute importance scores using a simpler but effective combination
    # We'll use a weighted sum of workload and variance
    workload_weight = 0.6
    variance_weight = 0.4
    
    # Normalize for fair weighting
    max_workload = np.max(job_workload) if np.max(job_workload) > 0 else 1.0
    max_variance = np.max(job_variance) if np.max(job_variance) > 0 else 1.0
    
    # Importance score: higher workload + higher variance = more important
    importance_scores = (
        workload_weight * (job_workload / max_workload) +
        variance_weight * (job_variance / max_variance)
    )
    
    # Step 2: Stochastic job selection with memory awareness
    # Select top jobs based on importance, but with some randomness to promote exploration
    num_perturb_jobs = 3  # Base number of jobs to perturb
    
    # Add randomness to selection process
    random_factor = 0.2  # 20% chance to include lower-ranked jobs
    
    # Create a probability distribution biased towards high importance but with some randomness
    probabilities = importance_scores.copy()
    
    # Apply softmax to create probabilities
    exp_probs = np.exp(probabilities * 3)  # Scale up to make differences more pronounced
    probabilities = exp_probs / np.sum(exp_probs)
    
    # Sample jobs with replacement based on probabilities
    selected_indices = np.random.choice(n, size=min(num_perturb_jobs, n), p=probabilities, replace=False)
    
    # Ensure we have at least 2 jobs
    if len(selected_indices) < 2:
        # Fill with highest variance jobs to ensure minimum count
        high_variance_indices = np.argsort(-job_variance)[:5]
        for idx in high_variance_indices:
            if idx not in selected_indices and len(selected_indices) < 5:
                selected_indices = np.append(selected_indices, idx)
    
    # Convert to actual job indices
    perturb_jobs = [current_sequence[i] for i in selected_indices[:5]]
    
    # Step 3: Apply perturbation with adaptive penalty factors
    # Base penalty that varies with iteration phase (more aggressive early on)
    base_penalty = 0.15
    
    # Apply penalties to selected jobs with variation based on job importance
    for job_idx in perturb_jobs:
        # Get the index in our job list
        job_position = current_sequence.index(job_idx)
        # Use importance score to determine penalty strength
        importance = importance_scores[job_position]
        # Make penalty stronger for more important jobs (but cap it)
        penalty_factor = base_penalty * (1.0 + 0.7 * importance)
        penalty_factor = min(penalty_factor, 0.5)  # Cap maximum penalty
        
        # Apply penalty to all machines for this job
        new_matrix[job_idx] = time_matrix[job_idx] * (1.0 + penalty_factor)
    
    # Step 4: Add exploration-inducing noise
    # Use a smaller noise level to maintain stability while encouraging exploration
    noise_level = 0.01  # Reduced from previous version for better stability
    random_noise = np.random.uniform(1 - noise_level, 1 + noise_level, size=(n, m))
    new_matrix = new_matrix * random_noise
    
    # Step 5: Ensure minimum processing times
    # Prevent negative or zero processing times
    new_matrix = np.maximum(new_matrix, time_matrix * 0.95)
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
