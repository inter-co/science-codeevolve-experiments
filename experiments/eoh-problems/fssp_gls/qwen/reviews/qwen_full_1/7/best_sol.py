# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using a novel hybrid approach combining machine analysis and temporal sampling.

    This implementation employs:
    1. Machine-based bottleneck identification using coefficient of variation
    2. Variance-based job prioritization with stochastic sampling
    3. Temporal correlation for local search effectiveness  
    4. Controlled perturbation with deterministic fallback

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
    
    # Strategy 1: Enhanced bottleneck identification using coefficient of variation
    machine_loads = np.sum(time_matrix, axis=0)
    
    # Use coefficient of variation to identify imbalanced machines
    machine_cv = np.std(machine_loads) / (np.mean(machine_loads) + 1e-8)
    
    # Select machines with high coefficient of variation (indicating imbalance)
    machine_means = np.mean(time_matrix, axis=0)
    machine_cvs = np.var(time_matrix, axis=0) / (machine_means + 1e-8)
    threshold_cv = np.percentile(machine_cvs, 70)
    bottleneck_machines = np.where(machine_cvs >= threshold_cv)[0]
    
    # Strategy 2: Job prioritization using variance-based stochastic sampling
    # Calculate job variance across machines to identify critical jobs
    job_variances = np.var(time_matrix, axis=1)
    
    # Normalize variances for probability calculation (handle edge case)
    if np.sum(job_variances) > 0:
        job_probabilities = job_variances / np.sum(job_variances)
    else:
        job_probabilities = np.ones(n) / n
    
    # Strategy 3: Hybrid job selection with temporal correlation
    selected_jobs = set()
    
    # Add jobs from bottleneck machines (if any) - prioritize these heavily
    if len(bottleneck_machines) > 0:
        # Collect all jobs that run on bottleneck machines
        bottleneck_jobs = set()
        for machine in bottleneck_machines:
            jobs_on_machine = np.where(time_matrix[:, machine] > 0)[0]
            bottleneck_jobs.update(jobs_on_machine)
        
        # Add bottleneck jobs with higher priority (up to 3)
        for job in list(bottleneck_jobs)[:3]:
            selected_jobs.add(job)
    
    # Add high-variance jobs for diversity - sample with probability proportional to variance
    num_variance_jobs = min(3, len(job_probabilities))
    sampled_variance_jobs = np.random.choice(n, size=num_variance_jobs, p=job_probabilities, replace=True)
    
    # Add unique sampled jobs
    for job in sampled_variance_jobs:
        if job not in selected_jobs:
            selected_jobs.add(job)
            if len(selected_jobs) >= 5:  # Early termination if we have enough
                break
    
    # Ensure we have at least 2 jobs (required by specification)
    if len(selected_jobs) < 2:
        # Add high-processing-time jobs as backup
        high_proc_jobs = np.argsort(-np.sum(time_matrix, axis=1))[:2]
        for job in high_proc_jobs:
            if job not in selected_jobs:
                selected_jobs.add(job)
                if len(selected_jobs) >= 2:
                    break
    
    # Convert to list and limit to 5 jobs
    perturb_jobs = list(selected_jobs)[:5]
    
    # Strategy 4: Apply structured perturbation
    # Base perturbation strength
    base_perturbation = 0.15
    
    # Create perturbation matrix
    perturbation_matrix = np.zeros_like(time_matrix)
    
    # Apply perturbation to selected jobs
    for job_idx in perturb_jobs:
        # Apply perturbation to all machines for this job
        for machine in range(m):
            if time_matrix[job_idx, machine] > 0:
                # Add perturbation with machine-specific scaling
                machine_factor = 1.0
                if machine in bottleneck_machines:
                    machine_factor = 1.5  # More aggressive perturbation on bottlenecks
                
                perturbation_amount = base_perturbation * machine_factor
                perturbation_matrix[job_idx, machine] = perturbation_amount
    
    # Strategy 5: Add temporal correlation for better local search
    # Include adjacent jobs in sequence to preserve temporal structure
    if len(perturb_jobs) > 0 and len(current_sequence) > 2:
        job_positions = {job: i for i, job in enumerate(current_sequence)}
        
        # Add adjacent jobs with higher probability to maintain sequence integrity
        additional_jobs = []
        for job in perturb_jobs:
            pos = job_positions.get(job, -1)
            if pos != -1:
                # Check neighbors in sequence with higher probability
                for offset in [-1, 1]:
                    neighbor_pos = pos + offset
                    if 0 <= neighbor_pos < len(current_sequence):
                        neighbor_job = current_sequence[neighbor_pos]
                        if neighbor_job not in perturb_jobs and np.random.random() < 0.4:
                            additional_jobs.append(neighbor_job)
        
        # Add any additional jobs without exceeding limit
        for job in additional_jobs:
            if len(perturb_jobs) < 5 and job not in perturb_jobs:
                perturb_jobs.append(job)
    
    # Final cleanup: ensure at least 2 jobs and remove duplicates
    if len(perturb_jobs) < 2:
        # Fallback to high processing time jobs
        high_proc_jobs = np.argsort(-np.sum(time_matrix, axis=1))[:2]
        for job in high_proc_jobs:
            if job not in perturb_jobs:
                perturb_jobs.append(job)
                if len(perturb_jobs) >= 2:
                    break
    
    # Limit to exactly 5 jobs max
    perturb_jobs = perturb_jobs[:5]
    
    # Apply final perturbation with noise
    noise = np.random.uniform(0.01, 0.05, size=time_matrix.shape)
    new_matrix = time_matrix * (1 + perturbation_matrix + noise)
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
