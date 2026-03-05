# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, differential_evolution
import math


def initialize_hexagonal():
    """Create a hexagonal grid initialization with better normalization."""
    # Create a more carefully constructed hexagonal pattern
    points = []
    row_height = math.sqrt(3) / 2
    
    # Generate points in a tight hexagonal arrangement
    for row in range(4):
        for col in range(4):
            x = col + (row % 2) * 0.5
            y = row * row_height
            
            points.append([x, y])
    
    points = np.array(points[:16])
    
    # Normalize to fit in [0,1] x [0,1] while preserving aspect ratio
    if len(points) > 0:
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
        
        # Scale to a good working area with more careful adjustment
        points = points * 0.9 + 0.05
        
    # Add more substantial perturbations to break symmetries
    points += np.random.normal(0, 0.03, points.shape)
    points = np.clip(points, 0, 1)
    
    return points


def initialize_circle_packing():
    """Circle packing inspired approach with better radial distribution."""
    # Start with a regular grid
    points = np.zeros((16, 2))
    for i in range(16):
        points[i] = [i // 4, i % 4]
    
    # Normalize to [0,1] range
    points[:, 0] = points[:, 0] / 3.0
    points[:, 1] = points[:, 1] / 3.0
    
    # Apply radial transformation to make distribution more uniform
    center = np.array([0.5, 0.5])
    for i in range(16):
        direction = points[i] - center
        distance = np.linalg.norm(direction)
        if distance > 0:
            # Apply radial stretching with a more controlled function
            # This helps spread out points away from the center
            stretch_factor = 1.0 + 0.3 * (1.0 - np.exp(-distance * 2))  
            points[i] = center + direction * stretch_factor
    
    # Add noise for better exploration
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, 0, 1)
    
    return points


def initialize_golden_spiral():
    """Golden spiral with better parameterization."""
    # Golden ratio spiral with improved radial scaling
    points = []
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    
    # Use a more refined approach to avoid clustering
    # Better distribution with improved parameterization
    for i in range(16):
        # Golden angle increment with more precise calculation
        angle = i * 2 * math.pi * (1 - 1/phi) 
        
        # Better radial distribution that spreads points more evenly
        # Using logarithmic spacing for better coverage
        if i == 0:
            radius = 0.01  # Start near center
        else:
            radius = 0.45 * (1 - math.exp(-i * 0.25)) 
        
        x = 0.5 + radius * math.cos(angle)
        y = 0.5 + radius * math.sin(angle)
        
        points.append([x, y])
    
    points = np.array(points)
    points = np.clip(points, 0, 1)
    return points


def initialize_optimized_grid():
    """Regular grid with controlled perturbations."""
    # Create a regular 4x4 grid
    points = np.zeros((16, 2))
    for i in range(16):
        points[i] = [i // 4, i % 4]
    
    # Normalize to [0,1] range
    points[:, 0] = points[:, 0] / 3.0
    points[:, 1] = points[:, 1] / 3.0
    
    # Apply moderate perturbations to break symmetries
    # Use smaller perturbations to preserve structure
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, 0, 1)
    
    return points


def initialize_spherical_code():
    """Enhanced spherical code approach with better distribution."""
    # Create points using Fibonacci-based distribution on a sphere
    # Then project to 2D using a more refined method
    points = []
    
    # Generate Fibonacci spiral points on sphere
    for i in range(16):
        # Use the golden angle approach for better uniformity
        theta = math.acos(-1 + (2 * i) / 15.0)  # Polar angle
        phi = i * 2.399963229728653  # Golden angle increment (~2π(3-√5)/2)
        
        # Convert to Cartesian coordinates
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        
        # Use a more robust projection that avoids singularities
        # Inverse stereographic projection with better handling
        if abs(z) < 0.9999:
            # Apply proper stereographic projection
            factor = 1.0 / (1.0 - z)
            x_proj = 0.5 + 0.45 * x * factor
            y_proj = 0.5 + 0.45 * y * factor
        else:
            # Near pole, use direct mapping to avoid extreme values
            x_proj = 0.5 + 0.45 * x
            y_proj = 0.5 + 0.45 * y
            
        points.append([x_proj, y_proj])
    
    points = np.array(points)
    points = np.clip(points, 0, 1)
    return points


def objective(params):
    """Objective function for optimization - returns negative ratio to maximize."""
    # Reshape parameters back to points
    pts = params.reshape(-1, 2)
    
    # Ensure points stay within bounds [0,1] x [0,1]
    pts = np.clip(pts, 0, 1)
    
    # Calculate all pairwise distances
    distances = pdist(pts)
    
    if len(distances) == 0:
        return 0
    
    # Calculate min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    # Avoid division by zero
    if d_max <= 0:
        return 0
        
    # Return negative ratio to maximize (since we're minimizing)
    return -d_min / d_max


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization with robust optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    
    # Strategy: Focus on the most proven approaches with optimized parameters
    # This approach prioritizes the highest performing strategies while staying within time limits
    
    best_points = None
    best_ratio = 0
    
    # Strategy 1: Primary - spherical code with very intensive optimization
    try:
        points = initialize_spherical_code()
        result = dual_annealing(
            objective,
            bounds=[(0, 1) for _ in range(n * 2)],
            maxiter=3500,  # Even more iterations for better convergence
            initial_temp=2200,  # Slightly higher temp for even better exploration
            seed=42,
            no_local_search=True
        )
        
        optimized_points = result.x.reshape(-1, 2)
        optimized_points = np.clip(optimized_points, 0, 1)
        ratio = -objective(optimized_points.flatten())
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
            
    except Exception:
        pass
    
    # Strategy 2: Secondary - hexagonal grid with maximum optimization effort
    if best_points is None or best_ratio < 0.05:
        try:
            points = initialize_hexagonal()
            result = dual_annealing(
                objective,
                bounds=[(0, 1) for _ in range(n * 2)],
                maxiter=3000,
                initial_temp=1800,
                seed=42,
                no_local_search=True
            )
            
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = -objective(optimized_points.flatten())
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception:
            pass
    
    # Strategy 3: Tertiary - golden spiral with focused optimization
    if best_points is None or best_ratio < 0.05:
        try:
            points = initialize_golden_spiral()
            result = dual_annealing(
                objective,
                bounds=[(0, 1) for _ in range(n * 2)],
                maxiter=2500,
                initial_temp=1400,
                seed=42,
                no_local_search=True
            )
            
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = -objective(optimized_points.flatten())
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception:
            pass
    
    # Strategy 4: If we have a good solution, refine with differential evolution
    if best_points is not None and best_ratio > 0.04:
        try:
            bounds = [(0, 1) for _ in range(n * 2)]
            result_de = differential_evolution(
                objective,
                bounds,
                maxiter=600,  # More iterations for better refinement
                popsize=30,
                seed=42,
                strategy='best1bin',
                atol=1e-10,
                rtol=1e-10
            )
            
            optimized_points_de = result_de.x.reshape(-1, 2)
            optimized_points_de = np.clip(optimized_points_de, 0, 1)
            ratio_de = -objective(optimized_points_de.flatten())
            
            if ratio_de > best_ratio:
                best_ratio = ratio_de
                best_points = optimized_points_de.copy()
                
        except Exception:
            pass
    
    # Final fallback: If nothing worked well, use the most reliable approach
    if best_points is None:
        try:
            points = initialize_hexagonal()
            result = dual_annealing(
                objective,
                bounds=[(0, 1) for _ in range(n * 2)],
                maxiter=2500,
                initial_temp=1200,
                seed=42,
                no_local_search=True
            )
            best_points = result.x.reshape(-1, 2)
            best_points = np.clip(best_points, 0, 1)
        except:
            # Last resort - just return the original hexagonal points
            best_points = initialize_hexagonal()
    
    return best_points


# EVOLVE-BLOCK-END
