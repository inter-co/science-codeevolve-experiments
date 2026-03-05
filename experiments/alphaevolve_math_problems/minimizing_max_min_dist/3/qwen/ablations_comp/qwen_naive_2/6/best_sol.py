# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
from scipy.spatial import ConvexHull
import warnings
from numba import jit
import random
from itertools import combinations


@jit(nopython=True)
def fast_pdist_jit(points):
    """Fast computation of pairwise distances using Numba"""
    n = points.shape[0]
    distances = np.zeros(n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = 0.0
            for k in range(points.shape[1]):
                diff = points[i, k] - points[j, k]
                dist += diff * diff
            distances[idx] = np.sqrt(dist)
            idx += 1
    return distances


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel geometric construction approach based on discrete optimization and symmetry principles.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    # Strategy: Geometric Construction Approach
    # Instead of optimization, construct points systematically using known optimal configurations
    # and then perform discrete refinement
    
    # Start with a known good configuration - icosahedral arrangement with additional points
    base_points = _construct_icosahedral_plus(n=14)
    
    # Apply discrete optimization approach: local search with neighbor swaps
    best_points = base_points.copy()
    best_ratio = _calculate_min_max_ratio(best_points)
    
    # Perform local search optimization
    improved = True
    max_iterations = 5000  # Prevent infinite loops
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try moving each point to several nearby positions
        for i in range(14):
            current_point = best_points[i].copy()
            current_ratio = best_ratio
            
            # Try several neighborhood moves
            for move_attempts in range(50):
                # Generate a small random displacement
                displacement = np.random.normal(0, 0.005, 3)
                new_point = current_point + displacement
                new_point = np.clip(new_point, 0, 1)
                
                # Test if this improves the ratio
                test_points = best_points.copy()
                test_points[i] = new_point
                
                new_ratio = _calculate_min_max_ratio(test_points)
                
                if new_ratio > current_ratio:
                    best_points[i] = new_point
                    best_ratio = new_ratio
                    improved = True
                    break
    
    # Final refinement with systematic grid search around best solution
    for _ in range(100):
        # Try to find even better point locations by sampling neighborhoods
        best_improvement = False
        for i in range(14):
            old_point = best_points[i].copy()
            old_ratio = best_ratio
            
            # Sample a few candidate positions around the current point
            for _ in range(20):
                # Small perturbations
                perturbation = np.random.normal(0, 0.001, 3)
                candidate = old_point + perturbation
                candidate = np.clip(candidate, 0, 1)
                
                test_points = best_points.copy()
                test_points[i] = candidate
                
                new_ratio = _calculate_min_max_ratio(test_points)
                if new_ratio > old_ratio:
                    best_points[i] = candidate
                    best_ratio = new_ratio
                    best_improvement = True
        
        if not best_improvement:
            break
    
    return best_points


def _construct_icosahedral_plus(n):
    """Construct an initial configuration based on icosahedral geometry plus additional points"""
    # Use vertices of regular icosahedron (12 vertices)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    
    # Normalize to unit sphere and map to [0,1]^3
    vertices = np.array(vertices)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    vertices = (vertices + 1) / 2  # map to [0,1]
    
    # Start with 12 icosahedral vertices
    points = vertices.copy()
    
    # Add 2 more points strategically
    # Place them at the center and one vertex position (to make it 14 total)
    points = np.vstack([points, [0.5, 0.5, 0.5]])  # center point
    points = np.vstack([points, vertices[0]])      # copy first vertex
    
    # Ensure we have exactly 14 points
    points = points[:14]
    
    # Add small random perturbations to avoid degenerate cases
    points += np.random.normal(0, 0.005, points.shape)
    points = np.clip(points, 0, 1)
    
    return points


def _calculate_min_max_ratio(points):
    """Helper function to calculate the min/max distance ratio"""
    if len(points) < 2:
        return 0
    
    # Use faster computation for large number of points
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist <= 0:
        return 0
    
    return min_dist / max_dist


# EVOLVE-BLOCK-END
