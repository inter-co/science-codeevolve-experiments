# EVOLVE-BLOCK-START

import numpy as np
from collections import deque

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using a hybrid approach combining game theory, temporal analysis, and 
    adaptive perturbation strategies for improved GLS performance.
    
    This implementation combines:
    1. Game-theoretic competitiveness with strategic sampling
    2. Temporal correlation analysis using sequence-aware features
    3. Adaptive penalty mechanisms with exploration control
    4. Diversity-enhanced job selection with temporal neighborhood consideration

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
    
    # Initialize memory for tracking perturbation effectiveness
    if not hasattr(get_matrix_and_jobs, 'perturbation_memory'):
        get_matrix_and_jobs.perturbation_memory = deque(maxlen=20)
    
    # Step 1: Multi-dimensional job characterization
    # Analyze jobs from multiple perspectives for comprehensive scoring
    
    # Processing time characteristics
    job_proc_times = np.sum(time_matrix, axis=1)
    max_proc = np.max(job_proc_times) if np.max(job_proc_times) > 0 else 1
    proc_normalized = job_proc_times / max_proc
    
    # Machine load impact analysis
    machine_loads = np.sum(time_matrix, axis=0)
    machine_avg = np.mean(machine_loads) if len(machine_loads) > 0 else 1
    machine_std = np.std(machine_loads) if len(machine_loads) > 1 else 0
    
    # Job-machine variance (flexibility measure)
    job_variance = np.var(time_matrix, axis=1)
    max_var = np.max(job_variance) if np.max(job_variance) > 0 else 1
    job_flexibility = job_variance / max_var
    
    # Sequence position analysis (early jobs are typically more critical)
    pos_analysis = np.array([1.0 - (i / max(1, n - 1)) for i in range(n)])
    
    # Step 2: Enhanced competitiveness scoring with balanced weighting
    # Create a composite score that balances multiple dimensions
    competitiveness = np.zeros(n)
    
    # Weighted combination of factors with careful normalization
    w1, w2, w3, w4 = 0.35, 0.25, 0.25, 0.15  # Adjusted weights for better balance
    
    # Factor 1: Processing time dominance (more important jobs get higher scores)
    competitiveness += w1 * proc_normalized
    
    # Factor 2: Machine load impact (jobs causing imbalances are more critical)
    machine_impact = np.zeros(n)
    for i in range(n):
        job_loads = time_matrix[i]
        if len(job_loads) > 0:
            # Calculate how much this job deviates from average machine load
            avg_load = np.mean(machine_loads)
            if avg_load > 0:
                # Use relative deviation instead of absolute to prevent extreme values
                job_impact = np.sum(np.abs(job_loads - avg_load)) / (avg_load + 1e-8)
                machine_impact[i] = min(1.0, job_impact)  # Clamp to reasonable range
    
    competitiveness += w2 * machine_impact
    
    # Factor 3: Flexibility (jobs with higher variance are more adaptable)
    competitiveness += w3 * job_flexibility
    
    # Factor 4: Positional importance (early jobs are more critical for makespan)
    competitiveness += w4 * pos_analysis
    
    # Step 3: Apply adaptive penalty scheme with improved exploration control
    # Use a dynamic temperature that decreases over time to shift from exploration to exploitation
    num_perturbations = len(get_matrix_and_jobs.perturbation_memory)
    temperature = max(0.1, 0.8 * (1.0 - num_perturbations / 50.0))  # Decreasing temperature
    
    # Apply penalties with controlled randomness
    penalty_factors = np.zeros(n)
    for i in range(n):
        # Base penalty based on competitiveness - clamp to prevent extreme values
        base_penalty = 0.1 * (1.0 + 0.7 * min(1.0, competitiveness[i]))
        
        # Add stochastic element for exploration
        if np.random.random() < 0.3:  # Reduced chance of randomness for stability
            # Add Gaussian noise scaled by competitiveness, with bounded effect
            noise = np.random.normal(0, 0.15 * (1.0 + competitiveness[i]))
            noise = np.clip(noise, -0.5, 0.5)  # Clamp noise to reasonable bounds
            penalty_factors[i] = base_penalty * (1.0 + noise)
        else:
            penalty_factors[i] = base_penalty
    
    # Apply penalties to all jobs
    for i in range(n):
        # Clamp penalty factors to prevent extreme perturbations
        penalty_factor = np.clip(penalty_factors[i], 0.0, 1.5)  # Prevent over-penalization
        new_matrix[i] = time_matrix[i] * (1.0 + penalty_factor)
    
    # Step 4: Strategic job selection with diversity enforcement
    # Select jobs using a combination of score-based sampling and diversity consideration
    
    # Calculate selection probability based on competitiveness
    # Apply softmax with temperature for more stable probability distribution
    # Clamp competitiveness values to prevent numerical overflow
    clamped_competitiveness = np.clip(competitiveness, -10, 10)
    sel_probs = np.exp(clamped_competitiveness / max(0.1, temperature))
    sel_probs = sel_probs / np.sum(sel_probs)
    
    # Select top jobs with probability proportional to competitiveness
    # but also ensure some diversity
    selected_jobs = []
    
    # Select 3-5 jobs based on probability distribution
    num_selected = min(5, max(2, n // 8 + 2))  # Slightly increased minimum
    
    # Sample with replacement to allow for better exploration
    sampled_indices = np.random.choice(n, size=num_selected, p=sel_probs, replace=False)
    
    # Convert back to actual job indices
    selected_jobs = [current_sequence[i] for i in sampled_indices]
    
    # Step 5: Enhance selection with temporal correlation
    # Add jobs that are temporally related to selected jobs for better cluster effects
    
    # Find neighbors in sequence for each selected job
    temporal_neighbors = set()
    for job_idx in selected_jobs:
        try:
            pos = current_sequence.index(job_idx)
            # Add immediate neighbors in sequence
            if pos > 0:
                temporal_neighbors.add(current_sequence[pos - 1])
            if pos < len(current_sequence) - 1:
                temporal_neighbors.add(current_sequence[pos + 1])
        except ValueError:
            continue
    
    # Add temporal neighbors that aren't already selected
    for neighbor in temporal_neighbors:
        if neighbor not in selected_jobs and len(selected_jobs) < 5:
            selected_jobs.append(neighbor)
    
    # Final validation to ensure minimum 2 jobs
    if len(selected_jobs) < 2:
        # Add jobs based on high processing times for additional diversity
        high_proc_indices = np.argsort(-job_proc_times)[:5]
        for idx in high_proc_indices:
            if idx not in selected_jobs and len(selected_jobs) < 5:
                # Convert back to job index in sequence
                job_idx = current_sequence[idx] if idx < len(current_sequence) else idx
                selected_jobs.append(job_idx)
    
    # Limit to exactly 5 jobs maximum
    perturb_jobs = selected_jobs[:5]
    
    # Ensure at least 2 jobs and remove duplicates
    if len(perturb_jobs) < 2:
        # Last resort: ensure we have at least 2 unique jobs
        unique_jobs = list(dict.fromkeys(perturb_jobs))
        while len(unique_jobs) < 2:
            # Add jobs from the beginning of the sequence
            for i in range(min(2, n)):
                if i not in unique_jobs:
                    unique_jobs.append(i)
                    if len(unique_jobs) >= 2:
                        break
        perturb_jobs = unique_jobs[:5]
    
    # Store this perturbation in memory for future learning
    get_matrix_and_jobs.perturbation_memory.append({
        'jobs': perturb_jobs.copy(),
        'sequence': current_sequence.copy()
    })
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
