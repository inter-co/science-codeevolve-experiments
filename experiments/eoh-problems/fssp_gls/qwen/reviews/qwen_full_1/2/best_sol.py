# EVOLVE-BLOCK-START

import numpy as np

def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    """Update the execution time matrix and select top jobs to perturb
    using a physics-inspired approach modeling job dynamics as particle motion
    in an energy landscape to escape local optima in GLS for FSSP.

    This implementation models the scheduling problem as a physical system where:
    1. Jobs are modeled as particles with kinetic and potential energy
    2. Energy landscape is computed based on current schedule quality
    3. Perturbations are generated using particle dynamics and force calculations
    4. Critical jobs are identified by energy gradient analysis

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
    
    # Phase 1: Physical System Modeling
    # Model jobs as particles in a 2D space where coordinates represent
    # their "position" in the schedule and "velocity" represents processing time variation
    
    # Compute job "kinetic energy" based on processing time variance across machines
    job_kinetic_energy = np.zeros(n)
    job_potential_energy = np.zeros(n)
    
    # Calculate variance in processing times for each job (higher variance = more dynamic)
    for i in range(n):
        if np.sum(time_matrix[i]) > 0:
            variance = np.var(time_matrix[i])
            job_kinetic_energy[i] = variance
        else:
            job_kinetic_energy[i] = 0
    
    # Compute potential energy based on job interactions (similar to electrostatics)
    # Jobs with similar processing patterns attract, different ones repel
    potential_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Cosine similarity for potential calculation
                dot_product = np.dot(time_matrix[i], time_matrix[j])
                norm_i = np.linalg.norm(time_matrix[i])
                norm_j = np.linalg.norm(time_matrix[j])
                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (norm_i * norm_j)
                    # Convert to potential energy (negative similarity = attraction)
                    potential_matrix[i, j] = -similarity
                else:
                    potential_matrix[i, j] = 0
    
    # Sum up potential energies for each job
    for i in range(n):
        job_potential_energy[i] = np.sum(np.abs(potential_matrix[i]))
    
    # Phase 2: Force Calculation and Dynamics Simulation
    # Calculate net forces acting on each job based on energy landscape
    # This simulates particle movement in the energy landscape
    
    # Net energy (total force) for each job
    job_net_energy = job_kinetic_energy + job_potential_energy
    
    # Normalize energies for comparison
    max_energy = np.max(job_net_energy) if np.max(job_net_energy) > 0 else 1
    normalized_energies = job_net_energy / max_energy
    
    # Phase 3: Particle-based Job Selection
    # Identify jobs that are "high energy" (critical) and "low energy" (stable)
    # High energy jobs are more likely to be in local optima and need perturbation
    
    # Sort jobs by net energy (descending)
    sorted_indices = np.argsort(-normalized_energies)
    
    # Select top jobs based on energy levels
    # Use a probabilistic approach to ensure diversity
    selected_indices = []
    
    # Select top 2-4 jobs based on highest energy
    num_selected = max(2, min(4, n // 4))  # At least 2, at most 4
    
    # Select high energy jobs
    for i in range(min(num_selected, len(sorted_indices))):
        selected_indices.append(sorted_indices[i])
    
    # Add some lower energy jobs for diversity (but still relevant)
    if len(selected_indices) < 5:
        # Add jobs that are not too low energy but also not too high
        for i in range(len(sorted_indices)):
            if len(selected_indices) >= 5:
                break
            idx = sorted_indices[i]
            if idx not in selected_indices:
                # Only add if energy level is reasonable (not too low)
                if normalized_energies[idx] > 0.2:
                    selected_indices.append(idx)
    
    # Ensure we have at least 2 jobs
    if len(selected_indices) < 2:
        # Fill with additional jobs based on processing time patterns
        # Add jobs with high variance in processing times
        variance_scores = np.var(time_matrix, axis=1)
        sorted_variance = np.argsort(-variance_scores)
        for i in range(len(sorted_variance)):
            if len(selected_indices) >= 5:
                break
            idx = sorted_variance[i]
            if idx not in selected_indices:
                selected_indices.append(idx)
    
    # Limit to 5 jobs maximum
    selected_indices = selected_indices[:5]
    
    # Convert to actual job indices from current sequence
    perturb_jobs = [current_sequence[i] for i in selected_indices if i < len(current_sequence)]
    
    # Ensure minimum of 2 jobs
    if len(perturb_jobs) < 2:
        # Add more jobs to meet minimum requirement
        remaining_indices = [i for i in range(len(current_sequence)) if current_sequence[i] not in perturb_jobs]
        for i in remaining_indices[:2-len(perturb_jobs)]:
            perturb_jobs.append(current_sequence[i])
    
    # Limit to 5 jobs maximum
    perturb_jobs = perturb_jobs[:5]
    
    # Phase 4: Physics-Inspired Perturbation Generation
    # Apply perturbations based on particle dynamics and force directions
    
    # Calculate force direction for each selected job
    # Jobs with higher net energy experience stronger "forces"
    job_forces = normalized_energies[selected_indices] 
    
    # Apply different types of perturbations based on job force magnitude
    for i, job_idx in enumerate(perturb_jobs):
        # Determine perturbation type based on force strength
        force_strength = job_forces[i] if i < len(job_forces) else 0.5
        
        # Base perturbation intensity
        base_intensity = 0.1 + 0.3 * force_strength
        
        # Apply time-based perturbations with physics-inspired scaling
        # Modify processing times based on force direction and job characteristics
        for machine in range(m):
            # Use sinusoidal modulation based on machine index and job force
            phase = (machine / m) * 2 * np.pi
            amplitude = base_intensity * (0.8 + 0.4 * np.sin(phase))
            
            # Apply perturbation with physics-inspired randomness
            perturbation_factor = 1.0 + amplitude * (np.random.rand() - 0.5) * 2
            
            # Ensure realistic bounds
            perturbation_factor = np.clip(perturbation_factor, 0.7, 1.3)
            
            # Apply to time matrix
            new_matrix[job_idx, machine] = time_matrix[job_idx, machine] * perturbation_factor
    
    # Phase 5: Structural Perturbations Based on Particle Motion
    # Introduce artificial "momentum" changes to encourage schedule restructuring
    
    # Add momentum-like effects - modify job processing times to simulate
    # velocity changes in particle motion
    momentum_effect = 0.05  # Small momentum effect
    
    # Apply to first few jobs to encourage reordering
    for i, job_idx in enumerate(perturb_jobs[:3]):
        # Apply momentum effect to last machine (most critical for schedule)
        if m > 0:
            # Add momentum-induced perturbation
            momentum_perturbation = 1.0 + momentum_effect * (np.random.rand() - 0.5) * 2
            momentum_perturbation = np.clip(momentum_perturbation, 0.9, 1.1)
            new_matrix[job_idx, m-1] = new_matrix[job_idx, m-1] * momentum_perturbation
    
    # Phase 6: Chaos-Based Diversification
    # Apply chaotic perturbations to ensure exploration of distant regions
    if len(perturb_jobs) > 0:
        # Apply chaotic perturbation to one random job among selected
        chaotic_job_idx = np.random.choice(perturb_jobs)
        
        # Use logistic map for chaotic behavior (x_n+1 = r*x_n*(1-x_n))
        # but in our case we'll use it for generating pseudo-random chaotic sequence
        chaotic_factor = 0.8 + 0.4 * np.random.rand()
        
        # Apply to random machine
        random_machine = np.random.randint(0, m)
        new_matrix[chaotic_job_idx, random_machine] = time_matrix[chaotic_job_idx, random_machine] * chaotic_factor
    
    return new_matrix, perturb_jobs
# EVOLVE-BLOCK-END
