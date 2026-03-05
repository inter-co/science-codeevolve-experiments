# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using novel direct structural modification and disruption-focused approach.

    This implementation focuses on:
    1. Direct structural modifications without complex scoring
    2. Statistical measures like processing time variance and machine utilization
    3. Strategic randomized selection biased toward disruption potential
    4. Penalty mechanisms emphasizing schedule disruption
    5. Simple heuristics for job selection

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
    
    # Strategy 1: Direct structural modification using adjacency-based approach
    # Modify processing times of jobs that are adjacent in current sequence
    # This creates direct disruption to the schedule structure
    
    # Calculate machine utilization ratios (how busy each machine is)
    machine_utilization = np.sum(time_matrix, axis=0) / np.max(np.sum(time_matrix, axis=0))
    
    # Strategy 2: Use statistical measures to identify high-variance jobs
    # Jobs with high processing time variance across machines are more disruptive
    job_processing_variances = np.var(time_matrix, axis=1)
    
    # Strategy 3: Identify bottleneck machines and their associated jobs
    # Jobs on heavily utilized machines are more likely to be part of bottlenecks
    bottleneck_threshold = 1.1  # machines above 110% of average utilization
    bottleneck_machines = np.where(machine_utilization > bottleneck_threshold)[0]
    
    # Strategy 4: Select jobs based on position in sequence (early jobs often more impactful)
    # and processing time variance (high variance jobs more disruptive)
    job_positions = np.array([current_sequence.index(i) for i in range(n)])
    
    # Create disruption potential scores based on simple heuristics
    disruption_scores = np.zeros(n)
    
    # High variance jobs (more likely to be disruptive when changed)
    disruption_scores += 0.4 * (job_processing_variances / np.max(job_processing_variances + 1e-8))
    
    # Position-based score (early jobs more impactful)
    disruption_scores += 0.3 * (1.0 - job_positions / (n - 1) if n > 1 else np.zeros(n))
    
    # Bottleneck machine score (jobs on busy machines more critical)
    bottleneck_score = np.zeros(n)
    for job_idx in range(n):
        # Count how many bottleneck machines this job is assigned to
        bottleneck_count = sum(1 for machine_idx in range(m) 
                              if time_matrix[job_idx, machine_idx] > 0 and machine_idx in bottleneck_machines)
        bottleneck_score[job_idx] = bottleneck_count / (len(bottleneck_machines) + 1e-8)
    disruption_scores += 0.3 * bottleneck_score
    
    # Strategy 5: Apply penalty mechanism that emphasizes disruption
    # Rather than just time scaling, apply penalties that create more significant schedule changes
    penalty_factor = 1.0 + 0.3 * (disruption_scores / np.max(disruption_scores + 1e-8))
    
    # Apply penalties with strategic randomness to ensure disruption
    for i in range(n):
        # Add some random noise to prevent over-penalization
        noise = 0.1 * np.random.random()  
        new_matrix[i] = time_matrix[i] * (penalty_factor[i] + noise)
    
    # Strategy 6: Select jobs for perturbation using a randomized approach with bias
    # Bias towards jobs with high disruption potential but with randomization for exploration
    
    # Sort by disruption scores but introduce randomness in selection
    sorted_indices = np.argsort(-disruption_scores)
    
    # Use a more diversified selection approach
    selected_jobs = []
    
    # Always include the highest scoring job
    if len(sorted_indices) > 0:
        selected_jobs.append(sorted_indices[0])
    
    # Select additional jobs with bias toward disruption scores but with randomization
    # This helps explore different parts of the search space
    remaining_candidates = [i for i in range(n) if i != sorted_indices[0]]
    
    # Add 2-4 more jobs based on disruption scores with randomization
    num_additional = np.random.randint(2, min(5, len(sorted_indices)))
    
    # Select based on a probabilistic approach that favors high-disruption jobs
    # but maintains some randomness
    for _ in range(num_additional):
        if len(remaining_candidates) == 0:
            break
            
        # Calculate probabilities proportional to disruption scores
        candidate_scores = [disruption_scores[i] for i in remaining_candidates]
        total_score = sum(candidate_scores)
        
        if total_score > 0:
            # Normalize scores to probabilities
            probs = [score / total_score for score in candidate_scores]
            # Select with probability proportional to disruption score
            chosen_idx = np.random.choice(remaining_candidates, p=probs)
        else:
            # If all scores are zero, select randomly
            chosen_idx = np.random.choice(remaining_candidates)
            
        selected_jobs.append(chosen_idx)
        remaining_candidates.remove(chosen_idx)
    
    # Ensure we have at least 2 jobs (required by specification)
    if len(selected_jobs) < 2:
        # Fill with random jobs that weren't already selected
        available = [i for i in range(n) if i not in selected_jobs]
        while len(selected_jobs) < 2 and available:
            selected_jobs.append(available.pop())
    
    # Final validation: ensure exactly 2-5 jobs (inclusive)
    selected_jobs = selected_jobs[:5]
    
    # Make sure we have at least 2 jobs
    if len(selected_jobs) < 2:
        # Add some random jobs to meet minimum requirement
        missing_count = 2 - len(selected_jobs)
        available = [i for i in range(n) if i not in selected_jobs]
        if len(available) >= missing_count:
            selected_jobs.extend(np.random.choice(available, size=missing_count, replace=False).tolist())
    
    return new_matrix, selected_jobs
# EVOLVE-BLOCK-END
