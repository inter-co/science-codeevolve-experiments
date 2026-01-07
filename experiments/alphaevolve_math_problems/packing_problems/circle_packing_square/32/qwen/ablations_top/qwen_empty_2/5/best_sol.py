# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import math
from typing import Tuple, List
import time

# Global constants for the optimization
MAX_ITERATIONS = 50000
TEMP_START = 1.0
TEMP_DECAY = 0.9995
MIN_TEMP = 1e-8

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining structured initialization, gradient-based refinement, 
    and simulated annealing.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Initialize with a structured configuration
    circles = initialize_structured_config(n)
    
    # Phase 2: Apply gradient-based refinement to improve initial solution
    circles = refine_with_gradient_descent(circles)
    
    # Phase 3: Refine with simulated annealing for global optimization
    circles = optimize_with_simulated_annealing(circles)
    
    # Phase 4: Final polishing with local search
    circles = polish_with_local_search(circles)
    
    return circles

def initialize_structured_config(n: int) -> np.ndarray:
    """Initialize circles using a structured approach"""
    circles = np.zeros((n, 3))
    
    # Determine grid dimensions
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Golden ratio spacing for better distribution
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Adjust spacing to ensure we don't exceed unit square
    max_spacing = min(spacing_x, spacing_y)
    actual_spacing_x = max_spacing * 0.95
    actual_spacing_y = max_spacing * 0.95
    
    # Place circles with some jitter for better distribution
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Add jitter to improve distribution
        jitter_x = (random.random() - 0.5) * actual_spacing_x * 0.2
        jitter_y = (random.random() - 0.5) * actual_spacing_y * 0.2
        
        # Position with offset for better packing
        x = (col + 0.5 + jitter_x / actual_spacing_x) * actual_spacing_x
        y = (row + 0.5 + jitter_y / actual_spacing_y) * actual_spacing_y
        
        # Ensure within bounds
        x = max(0.01, min(0.99, x))
        y = max(0.01, min(0.99, y))
        
        # Radius based on available space and grid size
        r = min(actual_spacing_x, actual_spacing_y) * 0.25
        
        circles[i] = [x, y, r]
    
    return circles

def refine_with_gradient_descent(circles: np.ndarray) -> np.ndarray:
    """Refine solution using gradient-based method with smooth constraint approximation"""
    n = len(circles)
    max_iter = 500
    learning_rate = 0.01
    
    # Convert to array for easier manipulation
    positions = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    
    # Smooth penalty function parameters
    penalty_weight = 1000.0
    overlap_smoothness = 0.01
    
    for iteration in range(max_iter):
        # Compute gradients
        grad_positions = np.zeros_like(positions)
        grad_radii = np.zeros_like(radii)
        
        # Compute forces from boundary constraints
        for i in range(n):
            # Boundary forces (soft constraints)
            boundary_force_x = 0
            boundary_force_y = 0
            
            if positions[i, 0] < radii[i]:
                boundary_force_x = penalty_weight * (radii[i] - positions[i, 0])
            elif positions[i, 0] > 1 - radii[i]:
                boundary_force_x = penalty_weight * (positions[i, 0] - (1 - radii[i]))
                
            if positions[i, 1] < radii[i]:
                boundary_force_y = penalty_weight * (radii[i] - positions[i, 1])
            elif positions[i, 1] > 1 - radii[i]:
                boundary_force_y = penalty_weight * (positions[i, 1] - (1 - radii[i]))
            
            grad_positions[i, 0] += boundary_force_x
            grad_positions[i, 1] += boundary_force_y
        
        # Compute forces from overlap constraints using smooth approximation
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Smooth approximation of overlap penalty
                overlap_distance = radii[i] + radii[j] - distance
                if overlap_distance > 0:
                    # Smooth penalty using exponential decay
                    penalty = penalty_weight * np.exp(-overlap_distance / overlap_smoothness)
                    
                    # Force direction
                    if distance > 1e-8:
                        force_x = penalty * dx / distance
                        force_y = penalty * dy / distance
                        
                        grad_positions[i, 0] -= force_x
                        grad_positions[i, 1] -= force_y
                        grad_positions[j, 0] += force_x
                        grad_positions[j, 1] += force_y
                        
                        # Also adjust radii to resolve overlaps
                        grad_radii[i] += penalty * 0.1
                        grad_radii[j] += penalty * 0.1
        
        # Update positions and radii
        positions -= learning_rate * grad_positions
        radii += learning_rate * grad_radii
        
        # Project back to valid domain
        for i in range(n):
            # Ensure circles stay within bounds
            positions[i, 0] = np.clip(positions[i, 0], radii[i], 1 - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], 1 - radii[i])
            radii[i] = np.maximum(radii[i], 0.001)
    
    # Update circles with refined values
    circles[:, :2] = positions
    circles[:, 2] = radii
    
    return circles

def polish_with_local_search(circles: np.ndarray) -> np.ndarray:
    """Apply local search to fine-tune the solution"""
    n = len(circles)
    
    # Create spatial index for efficient neighbor search
    tree = cKDTree(circles[:, :2])
    
    # Iteratively improve by local moves
    for _ in range(1000):  # Limited iterations for performance
        improved = False
        
        # Try moving each circle slightly
        for i in range(n):
            # Save current state
            old_pos = circles[i, :2].copy()
            old_rad = circles[i, 2]
            
            # Try small perturbations
            best_circles = circles.copy()
            best_sum = np.sum(circles[:, 2])
            
            for _ in range(10):  # Try 10 random perturbations
                # Small random move
                new_x = np.clip(old_pos[0] + np.random.normal(0, 0.001), old_rad, 1 - old_rad)
                new_y = np.clip(old_pos[1] + np.random.normal(0, 0.001), old_rad, 1 - old_rad)
                new_r = np.clip(old_rad + np.random.normal(0, 0.0005), 0.001, 0.5)
                
                # Test this configuration
                test_circles = circles.copy()
                test_circles[i, 0] = new_x
                test_circles[i, 1] = new_y
                test_circles[i, 2] = new_r
                
                # Check validity
                if is_valid_configuration(test_circles):
                    new_sum = np.sum(test_circles[:, 2])
                    if new_sum > best_sum:
                        best_circles = test_circles
                        best_sum = new_sum
                        improved = True
            
            circles = best_circles
            
            if not improved:
                break
    
    return circles

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if configuration is valid (no overlaps, within bounds)"""
    n = len(circles)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False
    
    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)
    
    # For each circle, check distance to all others
    for i in range(n):
        x1, y1, r1 = circles[i]
        # Find nearby circles (within 2*(r1+r2) distance)
        nearby = tree.query_ball_point([x1, y1], 2*(r1 + 0.01))
        
        for j in nearby:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < r1 + r2:
                    return False
    
    return True

def calculate_energy(circles: np.ndarray) -> float:
    """Calculate energy function that penalizes violations"""
    penalty = 0.0
    
    # Boundary penalties
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Penalize if circle extends beyond boundaries
        penalty += max(0, r - x)  # left boundary
        penalty += max(0, r - (1 - x))  # right boundary
        penalty += max(0, r - y)  # bottom boundary
        penalty += max(0, r - (1 - y))  # top boundary
    
    # Overlap penalties
    points = circles[:, :2]
    tree = cKDTree(points)
    
    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        nearby = tree.query_ball_point([x1, y1], 2*(r1 + 0.01))
        
        for j in nearby:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                overlap = max(0, r1 + r2 - distance)
                penalty += overlap * 1000  # Heavy penalty for overlaps
    
    return penalty

def perturb_circles(circles: np.ndarray, step_size: float = 0.01) -> np.ndarray:
    """Create a new configuration by slightly perturbing existing circles"""
    new_circles = circles.copy()
    
    # Choose a random circle to perturb
    idx = random.randint(0, len(new_circles) - 1)
    
    # Perturb position and radius
    x, y, r = new_circles[idx]
    
    # Random perturbation for position
    new_x = max(r, min(1-r, x + random.uniform(-step_size, step_size)))
    new_y = max(r, min(1-r, y + random.uniform(-step_size, step_size)))
    
    # Perturb radius (with bounds)
    new_r = max(0.001, min(0.5, r + random.uniform(-step_size/2, step_size/2)))
    
    new_circles[idx] = [new_x, new_y, new_r]
    
    return new_circles

def optimize_with_simulated_annealing(circles: np.ndarray) -> np.ndarray:
    """Optimize using simulated annealing"""
    current = circles.copy()
    best = circles.copy()
    best_energy = calculate_energy(current)
    best_radius_sum = np.sum(current[:, 2])
    
    temperature = TEMP_START
    
    for iteration in range(MAX_ITERATIONS):
        # Generate neighbor solution
        new_circles = perturb_circles(current, 0.01 * temperature)
        
        # Calculate energies
        current_energy = calculate_energy(current)
        new_energy = calculate_energy(new_circles)
        
        # Accept or reject based on Metropolis criterion
        if new_energy <= current_energy or random.random() < math.exp(-(new_energy - current_energy) / temperature):
            current = new_circles
            
            # Update best if improved
            new_radius_sum = np.sum(current[:, 2])
            if new_radius_sum > best_radius_sum:
                best = current.copy()
                best_radius_sum = new_radius_sum
                best_energy = new_energy
        
        # Cool down temperature
        temperature *= TEMP_DECAY
        
        # Early stopping condition
        if temperature < MIN_TEMP:
            break
    
    return best


# EVOLVE-BLOCK-END
