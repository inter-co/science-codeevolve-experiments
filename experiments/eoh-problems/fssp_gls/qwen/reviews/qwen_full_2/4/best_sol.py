# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using a hybrid variance-bottleneck-sequence approach for GLS in Flow Shop Scheduling.

    This implementation combines:
    - Variance-based sensitivity analysis (high variance jobs are more perturbable)
    - Bottleneck machine detection (jobs on congested machines need attention)
    - Sequence position awareness (diversity across schedule positions)
    - Multiplicative perturbations with bounded randomness

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
    
    # Step 1: Calculate job sensitivities using variance analysis
    # Jobs with high variance in processing times across machines are more sensitive to change
    job_variances = np.var(time_matrix, axis=1)
    
    # Step 2: Identify bottleneck machines using percentile-based threshold
    machine_loads = np.sum(time_matrix, axis=0)
    avg_load = np.mean(machine_loads)
    bottleneck_machines = np.where(machine_loads > avg_load * 1.2)[0]
    
    # Step 3: Calculate job importance scores using a balanced approach
    job_processing_totals = np.sum(time_matrix, axis=1)
    
    # Normalize scores for fair comparison
    normalized_variances = job_variances / (np.max(job_variances) + 1e-8)
    normalized_processing = job_processing_totals / (np.max(job_processing_totals) + 1e-8)
    
    # Calculate bottleneck bonus (jobs that use bottleneck machines)
    bottleneck_bonus = np.zeros(n)
    if len(bottleneck_machines) > 0:
        for machine in bottleneck_machines:
            bottleneck_bonus += (time_matrix[:, machine] > 0).astype(int)
        bottleneck_bonus = bottleneck_bonus / (np.max(bottleneck_bonus) + 1e-8)
    
    # Combined importance score with tuned weights
    importance_scores = (
        0.4 * normalized_variances +      # Sensitivity to changes (high variance jobs)
        0.4 * normalized_processing +     # Overall processing impact  
        0.2 * bottleneck_bonus            # Bottleneck criticality
    )
    
    # Step 4: Select top jobs for perturbation
    # Use a more deterministic approach with fixed percentage
    num_perturb_jobs = max(2, min(5, int(0.25 * n)))  # 25% of jobs
    top_importance_indices = np.argsort(-importance_scores)[:num_perturb_jobs]
    
    # Step 5: Apply adaptive perturbations to selected jobs
    for job_idx in top_importance_indices:
        # Base perturbation intensity based on job characteristics
        variance_factor = 1.0 + 0.2 * normalized_variances[job_idx]
        processing_factor = 0.5 + 0.5 * normalized_processing[job_idx]
        
        # Combined perturbation intensity (bounded between 15-30%)
        base_perturbation = 0.15 + 0.15 * np.random.random()
        perturbation = base_perturbation * variance_factor * processing_factor
        
        # Apply to all machines this job uses
        for machine in range(m):
            if time_matrix[job_idx, machine] > 0:
                # Apply multiplicative perturbation with bounds checking
                # Ensure we don't reduce processing times too much
                perturbation_amount = (np.random.random() - 0.5) * 2 * perturbation
                new_matrix[job_idx, machine] *= (1.0 + perturbation_amount)
                new_matrix[job_idx, machine] = max(1, new_matrix[job_idx, machine])
    
    # Step 6: Add diversity through sequence-based perturbations
    # Include jobs from early positions in sequence for temporal diversity
    early_sequence_jobs = []
    for i in range(min(2, len(current_sequence))):
        job_idx = current_sequence[i]
        if job_idx not in top_importance_indices and len(early_sequence_jobs) < 2:
            early_sequence_jobs.append(job_idx)
    
    # Apply small perturbations to early sequence jobs
    for job_idx in early_sequence_jobs:
        perturbation = 0.05 + 0.05 * np.random.random()  # 5-10%
        for machine in range(m):
            if time_matrix[job_idx, machine] > 0:
                new_matrix[job_idx, machine] *= (1.0 + perturbation)
                new_matrix[job_idx, machine] = max(1, new_matrix[job_idx, machine])
    
    # Step 7: Final job selection with diversity guarantees
    # Start with top importance jobs
    perturb_jobs = list(top_importance_indices)
    
    # Add early sequence jobs for positional diversity
    for job_idx in early_sequence_jobs:
        if job_idx not in perturb_jobs and len(perturb_jobs) < 5:
            perturb_jobs.append(job_idx)
    
    # Ensure we have at least 2 jobs and limit to 5
    perturb_jobs = list(dict.fromkeys(perturb_jobs))[:5]
    
    # Final safeguard: ensure at least 2 jobs
    if len(perturb_jobs) < 2:
        # Fill with additional jobs from sequence
        remaining_jobs = [i for i in range(n) if i not in perturb_jobs]
        while len(perturb_jobs) < 2 and remaining_jobs:
            perturb_jobs.append(remaining_jobs.pop())
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
