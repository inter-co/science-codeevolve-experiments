# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using a simple yet effective strategy for GLS in FSSP.

    This implementation uses:
    1. Processing time variance to identify problematic jobs
    2. Position-based selection to cover the entire sequence
    3. Simple penalty application for perturbation guidance
    4. Fast diversity selection without complex algorithms

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
    
    # Step 1: Calculate processing time variance for each job
    # Jobs with high variance are more sensitive to perturbations
    job_variances = np.var(time_matrix, axis=1)
    
    # Step 2: Identify jobs in critical positions in the sequence
    # Select jobs from beginning, middle, and end of sequence for coverage
    if len(current_sequence) >= 3:
        # Take jobs from first third, middle third, and last third
        seq_len = len(current_sequence)
        first_third = current_sequence[:seq_len//3]
        middle_third = current_sequence[seq_len//3:2*seq_len//3]
        last_third = current_sequence[2*seq_len//3:]
        
        # Get unique job indices from each section
        pos_jobs = set(first_third + middle_third + last_third)
    else:
        # If sequence too short, just take all jobs
        pos_jobs = set(current_sequence)
    
    # Step 3: Combine variance and positional importance
    # Create score that considers both processing time variance and position
    combined_scores = np.zeros(n)
    for i in range(n):
        # Variance-based importance (higher variance = more important)
        variance_score = job_variances[i]
        
        # Positional importance (jobs in critical positions get higher score)
        positional_score = 1.0 if i in pos_jobs else 0.0
        
        # Weighted combination
        combined_scores[i] = 0.7 * variance_score + 0.3 * positional_score
    
    # Step 4: Select top jobs for perturbation (limit to 5)
    # Sort by combined score and take top jobs
    top_job_indices = np.argsort(-combined_scores)[:min(5, n)]
    
    # Step 5: Apply penalties to selected jobs
    # Use simple multiplicative penalties based on variance
    base_penalty = 0.10  # Base 10% penalty
    max_penalty_increase = 0.20  # Up to 20% additional penalty
    
    for job_idx in top_job_indices:
        # Penalty based on variance - higher variance jobs get higher penalties
        variance_factor = job_variances[job_idx] / (np.max(job_variances) + 1e-8)
        penalty_strength = base_penalty + max_penalty_increase * variance_factor
        
        # Apply penalty to all machines for this job
        # Use small random variation to avoid getting stuck in same pattern
        penalty_multiplier = 1.0 + np.random.uniform(-penalty_strength, penalty_strength)
        new_matrix[job_idx] = time_matrix[job_idx] * penalty_multiplier
    
    # Step 6: Select final perturbation jobs ensuring minimum count
    # Start with top jobs, then add more if needed to reach minimum
    perturb_jobs = list(top_job_indices)
    
    # Ensure we have at least 2 jobs (required by specification)
    if len(perturb_jobs) < 2:
        # Add some additional jobs that are not in top list but have high variance
        remaining_indices = [i for i in range(n) if i not in perturb_jobs]
        additional_indices = np.argsort(-job_variances)[len(perturb_jobs):min(2, len(remaining_indices))]
        perturb_jobs.extend([remaining_indices[i] for i in additional_indices if i < len(remaining_indices)])
    
    # Limit to maximum 5 jobs
    perturb_jobs = perturb_jobs[:5]
    
    # Final safety check to ensure minimum requirement
    if len(perturb_jobs) < 2:
        # Fill with any other jobs
        available_jobs = [i for i in range(n) if i not in perturb_jobs]
        perturb_jobs.extend(available_jobs[:2-len(perturb_jobs)])
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
