# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    to escape local optima in Guided Local Search for Flow Shop Scheduling.

    This implementation uses a constraint satisfaction approach:
    - Identifies jobs that contribute most to scheduling conflicts
    - Applies perturbations that simulate conflict resolution
    - Uses game-theoretic principles to select jobs that would benefit most from repositioning

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
    
    # Step 1: Analyze constraint violations in the current schedule
    # Calculate job-machine conflict scores based on sequencing conflicts
    conflict_scores = np.zeros(n)
    
    # For each job, calculate how many machines it conflicts with in sequence
    # A conflict occurs when a job's processing time on a machine overlaps 
    # with another job's processing time on the same machine
    for job_idx in range(n):
        # Calculate completion time for this job in current sequence
        completion_times = np.zeros(m)
        for seq_idx, seq_job in enumerate(current_sequence):
            if seq_job == job_idx:
                # Compute completion times for this job
                if seq_idx == 0:
                    completion_times[0] = time_matrix[job_idx, 0]
                else:
                    prev_job = current_sequence[seq_idx - 1]
                    completion_times[0] = completion_times[0] + time_matrix[job_idx, 0]
                
                for machine_idx in range(1, m):
                    completion_times[machine_idx] = max(
                        completion_times[machine_idx-1],
                        completion_times[machine_idx-1] + time_matrix[job_idx, machine_idx-1]
                    ) + time_matrix[job_idx, machine_idx]
                
                # Calculate conflict score based on overlap with other jobs
                for other_job in range(n):
                    if other_job != job_idx:
                        # Check if this job has overlapping processing on any machine
                        for machine_idx in range(m):
                            if time_matrix[job_idx, machine_idx] > 0 and time_matrix[other_job, machine_idx] > 0:
                                conflict_scores[job_idx] += 1
                break
    
    # Step 2: Identify high-conflict jobs and high-value jobs
    # High processing time jobs (potential bottlenecks)
    job_totals = np.sum(time_matrix, axis=1)
    high_processing_jobs = np.where(job_totals > np.percentile(job_totals, 70))[0].tolist()
    
    # High variance jobs (inconsistent processing times)
    job_variances = np.var(time_matrix, axis=1)
    high_variance_jobs = np.where(job_variances > np.percentile(job_variances, 70))[0].tolist()
    
    # High conflict jobs (jobs causing scheduling issues)
    high_conflict_jobs = np.argsort(-conflict_scores)[:min(5, len(conflict_scores))].tolist()
    
    # Machine bottleneck analysis
    machine_loads = np.sum(time_matrix, axis=0)
    max_load = np.max(machine_loads)
    bottleneck_machines = np.where(machine_loads > 0.8 * max_load)[0].tolist()
    
    # Jobs running on bottleneck machines
    bottleneck_jobs = []
    for job_idx in range(n):
        for machine_idx in bottleneck_machines:
            if time_matrix[job_idx, machine_idx] > 0:
                bottleneck_jobs.append(job_idx)
                break
    
    # Combine all strategies to form candidate jobs
    candidate_jobs = set(high_processing_jobs + high_variance_jobs + high_conflict_jobs + bottleneck_jobs)
    candidate_jobs = list(candidate_jobs)
    
    # Ensure we have at least 2 candidates
    if len(candidate_jobs) < 2:
        # Add some random jobs to ensure diversity
        additional_jobs = np.random.choice(n, size=max(2 - len(candidate_jobs), 2), replace=False).tolist()
        candidate_jobs = list(set(candidate_jobs + additional_jobs))
    
    # Limit candidates to reasonable number
    if len(candidate_jobs) > 8:
        candidate_jobs = np.random.choice(candidate_jobs, size=8, replace=False).tolist()
    
    # Step 3: Select jobs to perturb (at least 2, up to 5)
    num_perturb_jobs = min(len(candidate_jobs), 5)
    if num_perturb_jobs < 2:
        # If we don't have enough candidates, sample from all jobs
        perturb_jobs = np.random.choice(n, size=2, replace=False).tolist()
    else:
        # Sample from candidates with preference for high-value jobs
        # We'll use a weighted selection to favor jobs with higher scores
        weights = np.ones(len(candidate_jobs))
        perturb_jobs = np.random.choice(candidate_jobs, size=num_perturb_jobs, replace=False).tolist()
    
    # Step 4: Apply structured perturbations
    # Perturbation based on constraint satisfaction principles
    for job_idx in perturb_jobs:
        # Strategy 1: Add multiplicative noise to processing times
        # This simulates uncertainty in processing times and helps escape local optima
        noise_factor = 1.0 + (np.random.random() - 0.5) * 0.3  # ±15% noise
        new_matrix[job_idx] = time_matrix[job_idx] * noise_factor
        
        # Strategy 2: Occasionally apply a "swap" operation between machines
        # This mimics a game-theoretic adjustment where jobs re-evaluate their machine assignment
        if np.random.random() < 0.3:  # 30% chance of swap
            # Find machines with non-zero processing times
            non_zero_machines = np.where(time_matrix[job_idx] > 0)[0]
            if len(non_zero_machines) >= 2:
                # Randomly select two machines to swap processing times
                machines_to_swap = np.random.choice(non_zero_machines, size=2, replace=False)
                temp = new_matrix[job_idx, machines_to_swap[0]]
                new_matrix[job_idx, machines_to_swap[0]] = new_matrix[job_idx, machines_to_swap[1]]
                new_matrix[job_idx, machines_to_swap[1]] = temp
    
    # Final safeguard: ensure we have at least 2 jobs
    if len(perturb_jobs) < 2:
        remaining_jobs = [j for j in range(n) if j not in perturb_jobs]
        if remaining_jobs:
            additional = np.random.choice(remaining_jobs, size=2 - len(perturb_jobs), replace=False).tolist()
            perturb_jobs.extend(additional)
    
    # Ensure exactly 5 jobs or fewer
    perturb_jobs = perturb_jobs[:5]
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
