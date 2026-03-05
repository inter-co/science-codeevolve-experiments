# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a hybrid approach combining mathematical construction, energy optimization, 
    and multi-start strategy to achieve high-quality point distributions efficiently.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance"""
        distances = pdist(points)
        if len(distances) == 0 or np.max(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    # More sophisticated mathematical construction based on algebraic number fields
    def construct_algebraic_points():
        """Construct points using algebraic number field theory for optimal distribution"""
        # Using roots of unity in complex plane and projecting to real 2D space
        # Based on the 16th roots of unity, which have excellent symmetry properties
        
        # Generate 16 points on unit circle
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Scale and translate to [0.1, 0.9] x [0.1, 0.9] to avoid boundary effects
        points = 0.4 * points + 0.5
        points = np.clip(points, 0.1, 0.9)
        
        # Add structured perturbations to break symmetries while preserving good distribution
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    # Energy-based model with improved mathematical foundation
    def energy_model(points, alpha=3.0):
        """Calculate total repulsive energy between points - improved version"""
        distances = pdist(points)
        # Avoid division by zero and penalize very small distances more strongly
        distances = np.maximum(distances, 1e-12)
        # Energy is sum of inverse distance powers (repulsive force) - higher alpha for stronger repulsion at short distances
        energy = np.sum(1.0 / (distances ** alpha))
        return energy
    
    def gradient_energy(points, alpha=3.0):
        """Compute gradient of energy with respect to point positions"""
        n = len(points)
        grad = np.zeros_like(points)
        
        # Vectorized computation for better performance
        for i in range(n):
            # Compute differences with all other points at once
            diff_vec = points[i] - points  # Shape (n, 2)
            dist_sq_vec = np.sum(diff_vec**2, axis=1)  # Shape (n,)
            
            # Mask out self-interaction
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            
            # Compute contributions only for non-self terms
            dist_sq = dist_sq_vec[mask]
            diff = diff_vec[mask]  # Shape (n-1, 2)
            
            # Avoid numerical issues
            dist_sq = np.maximum(dist_sq, 1e-12)
            dist = np.sqrt(dist_sq)
            
            # Gradient contribution: alpha * (1/dist^(alpha+2)) * diff
            factor = alpha * (1.0 / (dist**(alpha+2)))  # Shape (n-1,)
            grad[i] = np.sum(factor[:, np.newaxis] * diff, axis=0)
        
        return grad
    
    # Generate multiple high-quality initial configurations
    def generate_initial_configurations():
        """Generate several different initial configurations based on mathematical principles"""
        configs = []
        
        # Strategy 1: Algebraic construction based on roots of unity (improved)
        alg_points = construct_algebraic_points()
        configs.append(("algebraic", alg_points.copy()))
        
        # Strategy 2: Golden spiral with better radial distribution
        n = 16
        points = np.zeros((n, 2))
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        # Generate points using golden spiral with improved radial spacing
        for i in range(n):
            angle = 2 * np.pi * i / phi
            # Use more even radial distribution to avoid clustering
            radius = 0.4 * np.sqrt(i / (n - 1)) if i < n - 1 else 0.4
            points[i] = [
                0.5 + radius * np.cos(angle),
                0.5 + radius * np.sin(angle)
            ]
        
        # Add structured perturbations to improve initial spread
        np.random.seed(42)
        points += np.random.normal(0, 0.02, points.shape)
        points = np.clip(points, 0, 1)
        configs.append(("golden_spiral", points.copy()))
        
        # Strategy 3: Hexagonal lattice pattern with better spacing
        points_hex = np.zeros((16, 2))
        sqrt3 = np.sqrt(3)
        row_height = sqrt3 / 2
        col_spacing = 1.0
        row_spacing = row_height
        
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx >= 16:
                    break
                # Alternate column offset for hexagonal packing
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * row_spacing
                
                # Scale to fit nicely in [0,1] square with some margin
                points_hex[idx, 0] = 0.1 + 0.8 * x / (3.5 * col_spacing)
                points_hex[idx, 1] = 0.1 + 0.8 * y / (3.5 * row_spacing)
                idx += 1
        
        # Add small random perturbations to break symmetry
        np.random.seed(123)
        points_hex += np.random.normal(0, 0.015, (16, 2))
        points_hex = np.clip(points_hex, 0, 1)
        configs.append(("hexagonal", points_hex.copy()))
        
        # Strategy 4: 4x4 grid with perturbations
        grid_points = []
        for i in range(4):
            for j in range(4):
                # Use a slightly offset grid to avoid degenerate cases
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                grid_points.append([x, y])
        
        points_grid = np.array(grid_points)
        
        # Add structured perturbations to improve spread while maintaining good structure
        np.random.seed(456)
        # Add moderate noise to break symmetry but keep structure
        points_grid += np.random.normal(0, 0.02, points_grid.shape)
        
        # Clip to [0,1] bounds to ensure constraints are satisfied
        points_grid = np.clip(points_grid, 0, 1)
        configs.append(("grid", points_grid.copy()))
        
        # Strategy 5: Concentric rings with radial symmetry breaking
        points_ring = np.zeros((16, 2))
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.2, 0.4, 16)
        
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            points_ring[i] = [
                0.5 + radius * np.cos(angle),
                0.5 + radius * np.sin(angle)
            ]
        
        # Add perturbations to break symmetry
        np.random.seed(789)
        points_ring += np.random.normal(0, 0.015, points_ring.shape)
        points_ring = np.clip(points_ring, 0, 1)
        configs.append(("ring", points_ring.copy()))
        
        # Strategy 6: Random uniform with normalization
        np.random.seed(999)
        points_random = np.random.uniform(0, 1, (16, 2))
        configs.append(("random", points_random.copy()))
        
        return configs
    
    # Enhanced optimization with better convergence and adaptive parameters
    def optimize_with_adaptive_gradient_descent(points, max_iter=500, tolerance=1e-6):
        """Optimize point configuration using adaptive gradient descent"""
        current_points = points.copy()
        velocity = np.zeros_like(current_points)
        momentum = 0.9
        learning_rate = 0.01
        
        prev_energy = float('inf')
        patience_counter = 0
        patience_limit = 15
        
        # Adaptive learning rate adjustment
        lr_decay_factor = 0.995
        
        for iteration in range(max_iter):
            # Compute energy and gradient
            current_energy = energy_model(current_points, alpha=3.0)
            
            # Check for convergence
            if abs(prev_energy - current_energy) < tolerance:
                patience_counter += 1
                if patience_counter > patience_limit:
                    break
            else:
                patience_counter = 0
                
            prev_energy = current_energy
            
            # Compute gradient
            grad = gradient_energy(current_points, alpha=3.0)
            
            # Apply momentum and update
            velocity = momentum * velocity - learning_rate * grad
            current_points += velocity
            
            # Project back to valid bounds [0,1]
            current_points = np.clip(current_points, 0, 1)
            
            # Occasionally reproject to avoid drift and adjust learning rate
            if iteration % 50 == 0:
                current_points = np.clip(current_points, 0, 1)
                learning_rate *= lr_decay_factor  # Gradually decrease learning rate
                
        return current_points
    
    # Multi-start optimization with diverse strategies and better selection criteria
    def multi_start_optimization():
        """Run multiple optimizations from different starting points with robust error handling"""
        best_points = None
        best_ratio = 0
        
        # Generate multiple initial configurations
        initials = generate_initial_configurations()
        
        # Try each initial configuration with optimization
        for init_name, initial_points in initials:
            try:
                # Optimize using gradient descent
                optimized_points = optimize_with_adaptive_gradient_descent(
                    initial_points, 
                    max_iter=300,  # Reduced iterations for time constraint
                    tolerance=1e-5
                )
                
                # Evaluate quality
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            except Exception:
                continue
        
        # If no good solution found, use the best initial configuration
        if best_points is None:
            # Find the best among initial configurations
            best_initial_ratio = 0
            for _, initial_points in initials:
                ratio = compute_min_max_ratio(initial_points)
                if ratio > best_initial_ratio:
                    best_initial_ratio = ratio
                    best_points = initial_points.copy()
        
        return best_points if best_points is not None else generate_initial_configurations()[0][1]
    
    # Improved simulated annealing for final global refinement
    def simulated_annealing_refinement(initial_points):
        """Use simulated annealing for robust global optimization"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Simulated annealing parameters - tuned for better exploration and faster convergence
        temperature = 1.0
        cooling_rate = 0.995
        min_temperature = 1e-6
        iterations_per_temp = 100
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Annealing loop
        while temperature > min_temperature:
            for _ in range(iterations_per_temp):
                # Make small random perturbations
                new_points = current_points.copy()
                idx = np.random.randint(0, 16)
                # Use smaller perturbation for fine-tuning
                new_points[idx] += np.random.normal(0, 0.005, 2)
                new_points = np.clip(new_points, 0, 1)
                
                # Accept or reject based on energy change
                new_ratio = compute_min_max_ratio(new_points)
                
                if new_ratio > current_ratio:
                    current_points = new_points
                    current_ratio = new_ratio
                    
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = new_points.copy()
                else:
                    # Accept with probability based on temperature
                    delta = new_ratio - current_ratio
                    if np.random.random() < np.exp(delta / temperature):
                        current_points = new_points
                        current_ratio = new_ratio
            
            temperature *= cooling_rate
        
        return best_points
    
    # Main optimization pipeline
    try:
        # Run multi-start optimization to find good starting points
        points = multi_start_optimization()
        
        # Final refinement with simulated annealing
        refined_points = simulated_annealing_refinement(points)
        
        # Return the better of the two
        original_ratio = compute_min_max_ratio(points)
        refined_ratio = compute_min_max_ratio(refined_points)
        
        return refined_points if refined_ratio > original_ratio else points
        
    except Exception as e:
        # Fallback to algebraic construction with better parameters
        return construct_algebraic_points()


# EVOLVE-BLOCK-END
