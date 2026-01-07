# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
import warnings
import math
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import itertools
warnings.filterwarnings('ignore')

class MathematicalCirclePacker:
    """
    A mathematical optimization approach using constrained nonlinear programming
    to maximize sum of radii for circle packing in a unit square.
    """
    
    def __init__(self, n_circles=32):
        self.n_circles = n_circles
        self.bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n_circles
        # Precompute pairs of indices for constraint checking
        self.constraint_pairs = list(itertools.combinations(range(n_circles), 2))
        
    def objective(self, params):
        """Objective function: negative sum of radii (since we want to maximize)"""
        radii = params[2::3]  # Every third element starting from index 2
        return -np.sum(radii)
    
    def constraint_overlap(self, params, i, j):
        """Constraint that circles i and j don't overlap"""
        x1, y1, r1 = params[3*i:3*i+3]
        x2, y2, r2 = params[3*j:3*j+3]
        
        # Distance between centers
        dx = x1 - x2
        dy = y1 - y2
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Should be greater than or equal to sum of radii
        return distance - (r1 + r2)
    
    def constraint_containment(self, params, i):
        """Constraint that circle i is fully contained"""
        x, y, r = params[3*i:3*i+3]
        # Circle must fit within unit square
        return min(x - r, 1 - x - r, y - r, 1 - y - r)
    
    def setup_constraints(self):
        """Setup all constraints for optimization"""
        constraints = []
        
        # Add containment constraints for all circles
        for i in range(self.n_circles):
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, idx=i: self.constraint_containment(params, idx)
            })
        
        # Add overlap constraints for all pairs
        for i, j in self.constraint_pairs:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda params, idx1=i, idx2=j: self.constraint_overlap(params, idx1, idx2)
            })
            
        return constraints
    
    def initialize_from_grid(self):
        """Initialize positions using a structured grid approach"""
        # Create a better initial configuration using a modified grid
        positions = []
        rows = int(np.ceil(np.sqrt(self.n_circles)))
        cols = int(np.ceil(self.n_circles / rows))
        
        # Use a more uniform grid with spacing adjustments
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        offset_x = 0.05
        offset_y = 0.05
        
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= self.n_circles:
                    break
                x = offset_x + j * spacing_x
                y = offset_y + i * spacing_y
                # Add small jitter to avoid perfect patterns
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                positions.append([x, y])
        
        # Fill remaining positions randomly
        while len(positions) < self.n_circles:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            positions.append([x, y])
        
        # Create initial parameter vector [x1, y1, r1, x2, y2, r2, ...]
        params = []
        for i, (x, y) in enumerate(positions[:self.n_circles]):
            # Start with small radii that can grow
            r = min(0.03, 0.5 * min(x, 1-x, y, 1-y))
            params.extend([x, y, r])
            
        return np.array(params)
    
    def optimize(self, maxiter=1000):
        """Run mathematical optimization with appropriate parameters"""
        # Initialize with good starting point
        x0 = self.initialize_from_grid()
        
        # Setup constraints
        constraints = self.setup_constraints()
        
        # Optimization options
        options = {
            'maxiter': maxiter,
            'ftol': 1e-6,
            'gtol': 1e-6,
            'disp': False
        }
        
        try:
            # Use SLSQP optimizer which handles constraints well
            result = minimize(
                self.objective,
                x0,
                method='SLSQP',
                bounds=self.bounds,
                constraints=constraints,
                options=options,
                tol=1e-6
            )
            
            if result.success:
                return result.x
            else:
                # If optimization fails, return initial configuration
                return x0
                
        except Exception as e:
            # Fallback to initial configuration if optimization fails
            return x0

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize sum of radii.
    Uses a mathematical programming approach with constrained nonlinear optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create mathematical packer
    packer = MathematicalCirclePacker(n_circles=32)
    
    # Run optimization
    start_time = time.time()
    params = packer.optimize(maxiter=500)
    end_time = time.time()
    
    # Convert parameters back to circles array
    circles = []
    for i in range(32):
        x, y, r = params[3*i:3*i+3]
        circles.append([x, y, r])
    
    circles = np.array(circles)
    
    # Post-process to ensure constraints are met
    circles = enforce_constraints(circles)
    
    # Validate final configuration
    if not validate_solution(circles):
        # If validation fails, return a fallback configuration
        # Create a better grid-based configuration
        fallback_circles = []
        rows = int(np.ceil(np.sqrt(32)))
        cols = int(np.ceil(32 / rows))
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        for i in range(32):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * spacing_x
            y = (row + 0.5) * spacing_y
            r = 0.03  # Small radius for fallback
            fallback_circles.append([x, y, r])
        
        return np.array(fallback_circles)
    
    return circles

def enforce_constraints(circles):
    """Ensure all constraints are satisfied with improved robustness"""
    # Make sure all circles are within bounds and have valid radii
    for i in range(len(circles)):
        x, y, r = circles[i]
        
        # Ensure containment
        max_radius = min(x, 1-x, y, 1-y)
        r = min(r, max_radius)
        r = max(r, 0.001)  # Prevent zero radius
        
        # Ensure bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        
        circles[i] = [x, y, r]
    
    return circles

def validate_solution(circles):
    """Validate that solution satisfies all constraints with better checking"""
    # Check containment
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlaps with more precise tolerance
    positions = circles[:, :2]
    radii = circles[:, 2]
    distances = cdist(positions, positions)
    
    # Use a small tolerance for floating point comparison
    tolerance = 1e-10
    
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            distance = distances[i, j]
            min_distance = radii[i] + radii[j]
            if distance < min_distance - tolerance:
                return False
                
    return True


# EVOLVE-BLOCK-END
