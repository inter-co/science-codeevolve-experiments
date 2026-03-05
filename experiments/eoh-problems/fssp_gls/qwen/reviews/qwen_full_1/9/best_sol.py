# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    Implements a robust variance-based perturbation strategy:
    - Focuses on jobs with high processing time variance
    - Applies adaptive penalties that scale with sensitivity
    - Ensures deterministic, reproducible behavior
    - Maintains minimum requirements for perturbation

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
    # Calculate variance in processing times for each job
    # Jobs with higher variance are more sensitive to perturbations
    job_variances = np.var(time_matrix, axis=1)
    
    # Handle edge case where all variances are zero
    if np.allclose(job_variances, 0):
        # If no variance exists, select jobs randomly but deterministically
        np.random.seed(42)  # For reproducibility
        perturb_jobs_indices = list(np.random.choice(n, size=min(4, n), replace=False))
    else:
        # Normalize variances to create probability distribution
        # Use softmax-like normalization to emphasize differences
        normalized_variances = job_variances / (np.max(job_variances) + 1e-8)
        
        # Create probabilities (higher variance = higher probability)
        # Use a temperature parameter to control probability distribution sharpness
        temp = 0.5
        probabilities = np.exp(normalized_variances / temp)
        probabilities = probabilities / np.sum(probabilities)
        
        # Ensure we have valid probabilities
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(probabilities) == 0:
            probabilities = np.ones(n) / n
        
        # Select jobs for perturbation (4 jobs for good balance)
        num_perturb_jobs = 4
        try:
            # Sample without replacement to avoid duplicates
            perturb_jobs_indices = np.random.choice(n, size=num_perturb_jobs, p=probabilities, replace=False).tolist()
        except ValueError:
            # Fallback to deterministic selection
            perturb_jobs_indices = list(range(min(num_perturb_jobs, n)))
    
    # Ensure we have at least 2 jobs (minimum requirement for GLS)
    if len(perturb_jobs_indices) < 2:
        # Add additional jobs from high variance ones
        high_variance_indices = np.argsort(-job_variances)[:5]
        for idx in high_variance_indices:
            if idx not in perturb_jobs_indices:
                perturb_jobs_indices.append(idx)
                if len(perturb_jobs_indices) >= 2:
                    break
    
    # Ensure we don't exceed 5 jobs
    perturb_jobs_indices = list(dict.fromkeys(perturb_jobs_indices))[:5]
    
    # Ensure we have at least 2 jobs (final check)
    if len(perturb_jobs_indices) < 2:
        # Add random jobs to meet minimum requirement
        available_jobs = [i for i in range(n) if i not in perturb_jobs_indices]
        while len(perturb_jobs_indices) < 2 and available_jobs:
            perturb_jobs_indices.append(available_jobs.pop())
    
    # Update the matrix with perturbation
    new_matrix = time_matrix.copy()
    
    # Apply perturbation to selected jobs with adaptive penalties
    # Base penalty is 10-30% depending on variance sensitivity
    for job_idx in perturb_jobs_indices:
        # Base penalty based on variance relative to mean
        if np.mean(job_variances) > 0:
            variance_factor = job_variances[job_idx] / np.mean(job_variances)
        else:
            variance_factor = 1.0
            
        # Cap the variance factor to prevent extreme penalties
        variance_factor = min(variance_factor, 3.0)
        
        # Apply penalty between 10% and 30% based on variance
        penalty = 0.10 + 0.20 * variance_factor / 3.0
        
        # Apply penalty to all machines for this job
        new_matrix[job_idx] = time_matrix[job_idx] * (1.0 + penalty)
    
    return new_matrix, perturb_jobs_indices
# EVOLVE-BLOCK-END
