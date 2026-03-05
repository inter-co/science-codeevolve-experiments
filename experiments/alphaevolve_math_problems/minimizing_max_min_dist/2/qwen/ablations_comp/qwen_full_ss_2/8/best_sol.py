# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a streamlined approach combining geometric initialization with efficient optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    
    # Simpler and cleaner hexagonal grid approach (from INSPIRATION 1)
    def initialize_hexagonal():
        """Create a clean 4x4 hexagonal grid pattern."""
        points = []
        # Create a regular grid with hexagonal offset
        for i in range(4):
            for j in range(4):
                x = j + (i % 2) * 0.5
                y = i * math.sqrt(3)/2
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize to [0.1, 0.9] x [0.1, 0.9] 
        if points.shape[0] > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range * 0.8 + 0.1
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range * 0.8 + 0.1
        
        return points
    
    # Golden ratio spiral approach (from INSPIRATION 1)
    def initialize_golden_spiral():
        """Golden ratio spiral approach."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        # Distribute points using golden angle with improved radial spacing
        for i in range(n):
            # Use golden angle increment (more precise)
            angle = i * 2 * math.pi * (1 - 1/phi)  
            
            # Better radial distribution to avoid clustering at center
            radius = 0.4 * math.sqrt(i / float(max(1, n - 1)))  # sqrt scaling
            
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            
            points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Objective function for optimization (from INSPIRATION 1)
    def objective(params):
        # Reshape parameters back to points
        pts = params.reshape(-1, 2)
        
        # Ensure points stay within bounds [0,1] x [0,1]
        pts = np.clip(pts, 0, 1)
        
        # Calculate all pairwise distances
        distances = pdist(pts)
        
        if len(distances) == 0:
            return -1e-10
        
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero - return small negative value instead
        if d_max <= 0:
            return -1e-10
            
        # Return negative ratio to maximize (since we're minimizing)
        return -d_min / d_max
    
    # Strategy: Follow INSPIRATION 1 approach with better parameters
    # Primary approach: Hexagonal grid with more iterations and higher temperature
    try:
        points = initialize_hexagonal()
        result = dual_annealing(
            objective,
            bounds=[(0, 1) for _ in range(n * 2)],
            maxiter=1500,  # More iterations for better convergence (as per inspiration)
            initial_temp=600,  # Higher temperature for better exploration (as per inspiration)
            seed=42,
            no_local_search=True
        )
        
        optimized_points = result.x.reshape(-1, 2)
        optimized_points = np.clip(optimized_points, 0, 1)
        ratio = -objective(optimized_points.flatten())
        
        best_points = optimized_points.copy()
        best_ratio = ratio
        
    except Exception:
        # Fallback to golden spiral approach with moderate settings
        try:
            points = initialize_golden_spiral()
            result = dual_annealing(
                objective,
                bounds=[(0, 1) for _ in range(n * 2)],
                maxiter=1000,  # Moderate iterations
                initial_temp=400,  # Moderate temperature
                seed=42,
                no_local_search=True
            )
            
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = -objective(optimized_points.flatten())
            
            best_points = optimized_points.copy()
            best_ratio = ratio
            
        except Exception:
            # Final fallback to hexagonal grid with standard optimization
            points = initialize_hexagonal()
            result = dual_annealing(
                objective,
                bounds=[(0, 1) for _ in range(n * 2)],
                maxiter=800,
                initial_temp=300,
                seed=42,
                no_local_search=True
            )
            
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = -objective(optimized_points.flatten())
            
            best_points = optimized_points
    
    return best_points


# EVOLVE-BLOCK-END
