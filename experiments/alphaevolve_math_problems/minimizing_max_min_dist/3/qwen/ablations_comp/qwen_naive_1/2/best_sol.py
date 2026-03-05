# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import time
from numba import jit
import math
from scipy.spatial import ConvexHull
import random


@jit(nopython=True)
def compute_min_max_ratio_jit(points):
    """Compute min/max distance ratio using numba for speed"""
    n = points.shape[0]
    min_dist_sq = np.inf
    max_dist_sq = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2 + (points[i,2]-points[j,2])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
    
    if max_dist_sq == 0:
        return 0.0
    return np.sqrt(min_dist_sq) / np.sqrt(max_dist_sq)


def generate_initial_config():
    """Generate a better initial configuration using spherical code principles"""
    # Use a more systematic approach for initial placement
    # Try to distribute points more evenly using Fibonacci-like distribution on sphere
    points = []
    
    # Generate points using fibonacci sphere method
    for i in range(14):
        y = 1 - (i / (14 - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = np.arccos(y)  # angle from z-axis
        phi_angle = (i * 4.0 * np.pi) / (14 - 1)  # angle around z-axis (modified for better distribution)
        
        x = radius * np.cos(phi_angle)
        z = radius * np.sin(phi_angle)
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / np.max(norms)
    
    # Scale to fit within [0,1]^3
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    
    # Normalize to [0,1]^3
    points = (points - mins) / (maxs - mins + 1e-10)
    points = np.clip(points, 0, 1)
    
    # Add small random perturbations to break symmetries
    points += np.random.normal(0, 0.005, points.shape)
    points = np.clip(points, 0, 1)
    
    return points


def energy_based_optimization(points, max_iter=5000):
    """Improved optimization using energy minimization with better force model"""
    # Physics parameters
    dt = 0.0005  # Reduced time step for better stability
    friction = 0.95  # Increased friction
    force_strength = 1.0
    
    # Initialize velocities
    velocities = np.zeros_like(points)
    
    best_points = points.copy()
    best_ratio = compute_min_max_ratio_jit(points)
    
    for step in range(max_iter):
        # Compute forces using a softened repulsion force
        forces = np.zeros_like(points)
        
        # Vectorized force computation
        for i in range(14):
            for j in range(i+1, 14):
                # Vector from i to j
                r_vec = points[j] - points[i]
                r = np.linalg.norm(r_vec)
                
                if r > 1e-10:  # Avoid division by zero
                    # Softened repulsion force (inverse cube to prevent extreme clustering)
                    force_magnitude = force_strength / (r * r * r + 1e-12)
                    force_vector = force_magnitude * r_vec / r
                    forces[i] += force_vector
                    forces[j] -= force_vector  # Newton's third law
        
        # Apply forces to velocities
        velocities += forces * dt
        
        # Apply friction
        velocities *= friction
        
        # Add small thermal noise for exploration
        velocities += np.random.normal(0, 0.001, velocities.shape)
        
        # Update positions
        points += velocities * dt
        
        # Boundary constraints with reflection
        for i in range(14):
            for dim in range(3):
                if points[i, dim] < 0:
                    points[i, dim] = -points[i, dim]  # Reflect
                    velocities[i, dim] *= -0.3  # Reverse velocity with damping
                elif points[i, dim] > 1:
                    points[i, dim] = 2 - points[i, dim]  # Reflect
                    velocities[i, dim] *= -0.3  # Reverse velocity with damping
        
        # Keep points within bounds
        points = np.clip(points, 0, 1)
        
        # Track best solution
        current_ratio = compute_min_max_ratio_jit(points)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
        
        # Early stopping conditions
        if step > 1000 and abs(np.mean(np.abs(forces))) < 1e-7:
            break
            
        # Occasionally re-scale to keep points within bounds
        if step % 100 == 0:
            points = np.clip(points, 0, 1)
    
    return best_points


def local_refinement(points, iterations=2000):
    """Use a more sophisticated local refinement approach"""
    def objective(x):
        points_test = x.reshape(-1, 3)
        try:
            ratio = compute_min_max_ratio_jit(points_test)
            return -ratio  # Negative because we want to maximize
        except:
            return -1e-10
    
    # Use differential evolution for global search first
    bounds = [(0, 1) for _ in range(42)]
    
    # Use a hybrid approach: start with DE then local optimization
    result = differential_evolution(objective, bounds, maxiter=50, popsize=15, seed=42)
    
    # Then refine locally with gradient-based method
    x_current = result.x.copy()
    
    # Simple gradient ascent with adaptive step size
    for iteration in range(iterations):
        epsilon = 1e-6
        grad = np.zeros_like(x_current)
        
        # Compute gradient numerically
        for i in range(len(x_current)):
            x_plus = x_current.copy()
            x_plus[i] += epsilon
            x_minus = x_current.copy()
            x_minus[i] -= epsilon
            
            val_plus = -objective(x_plus)
            val_minus = -objective(x_minus)
            grad[i] = (val_plus - val_minus) / (2 * epsilon)
        
        # Adaptive step size based on gradient magnitude
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            step_size = min(0.01 / grad_norm, 0.01)
            x_current += step_size * grad
        else:
            break
            
        x_current = np.clip(x_current, 0, 1)
    
    return x_current.reshape(-1, 3)


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses improved optimization techniques including better initialization, energy-based optimization,
    and local refinement.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate initial configuration
    points = generate_initial_config()
    
    # Run energy-based optimization
    optimized_points = energy_based_optimization(points)
    
    # Local refinement
    final_points = local_refinement(optimized_points)
    
    return final_points


# EVOLVE-BLOCK-END
