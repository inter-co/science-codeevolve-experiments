# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

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
    
    # Energy-based model inspired by physics and mathematics
    def energy_model(points, alpha=2.0):
        """Calculate total repulsive energy between points"""
        distances = pdist(points)
        # Avoid division by zero and penalize very small distances
        distances = np.maximum(distances, 1e-10)
        # Energy is sum of inverse distance powers (repulsive force)
        energy = np.sum(1.0 / (distances ** alpha))
        return energy
    
    def gradient_energy(points, alpha=2.0):
        """Compute gradient of energy with respect to point positions"""
        n = len(points)
        grad = np.zeros_like(points)
        
        # For each point, compute contribution to gradient from all other points
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq > 1e-12:  # Avoid numerical issues
                        dist = np.sqrt(dist_sq)
                        # Gradient of 1/dist^alpha w.r.t. point i
                        factor = alpha * (1.0 / (dist**(alpha+2))) 
                        grad[i] += factor * diff
        
        return grad
    
    # Generate high-quality initial configurations based on mathematical principles
    def generate_initial_configurations():
        """Generate several different initial configurations based on mathematical principles"""
        configs = []
        
        # Strategy 1: Golden spiral with better radial distribution
        n = 16
        points = np.zeros((n, 2))
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        # Generate points using golden spiral with optimized parameters
        for i in range(n):
            angle = 2 * np.pi * i / phi
            # Use more even radial distribution with slightly different scaling
            radius = 0.4 * np.sqrt(i / (n - 1)) if i < n - 1 else 0.4
            points[i] = [
                0.5 + radius * np.cos(angle),
                0.5 + radius * np.sin(angle)
            ]
        
        # Add structured perturbations to improve initial spread
        np.random.seed(42)
        points += np.random.normal(0, 0.012, points.shape)
        points = np.clip(points, 0, 1)
        configs.append(("golden_spiral", points.copy()))
        
        # Strategy 2: Hexagonal lattice pattern with better spacing
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
        points_hex += np.random.normal(0, 0.012, (16, 2))
        points_hex = np.clip(points_hex, 0, 1)
        configs.append(("hexagonal", points_hex.copy()))
        
        # Strategy 3: Grid with improved perturbations
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
        points_grid += np.random.normal(0, 0.012, points_grid.shape)
        
        # Clip to [0,1] bounds to ensure constraints are satisfied
        points_grid = np.clip(points_grid, 0, 1)
        configs.append(("grid", points_grid.copy()))
        
        # Strategy 4: Concentric rings with radial symmetry breaking
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
        points_ring += np.random.normal(0, 0.012, points_ring.shape)
        points_ring = np.clip(points_ring, 0, 1)
        configs.append(("ring", points_ring.copy()))
        
        # Strategy 5: Circle arrangement with mixed radii (better coverage)
        points_circle = np.zeros((16, 2))
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        # Mix inner and outer ring points for better coverage
        radii_inner = np.linspace(0.25, 0.35, 8)
        radii_outer = np.linspace(0.45, 0.55, 8)
        radii = np.concatenate([radii_inner, radii_outer])
        
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            points_circle[i] = [
                0.5 + radius * np.cos(angle),
                0.5 + radius * np.sin(angle)
            ]
        
        # Add perturbations
        np.random.seed(999)
        points_circle += np.random.normal(0, 0.012, points_circle.shape)
        points_circle = np.clip(points_circle, 0, 1)
        configs.append(("circle", points_circle.copy()))
        
        # Strategy 6: Random uniform distribution
        np.random.seed(1000)
        points_random = np.random.uniform(0, 1, (16, 2))
        configs.append(("random", points_random.copy()))
        
        # Strategy 7: Special configuration inspired by mathematical optimality
        # This is a more evenly distributed configuration
        special_points = np.zeros((16, 2))
        # Create a pattern that avoids clustering
        for i in range(4):
            for j in range(4):
                special_points[i*4 + j] = [
                    0.1 + 0.8 * i / 3.0,
                    0.1 + 0.8 * j / 3.0
                ]
        # Add small perturbations
        np.random.seed(1111)
        special_points += np.random.normal(0, 0.01, special_points.shape)
        special_points = np.clip(special_points, 0, 1)
        configs.append(("special", special_points.copy()))
        
        return configs
    
    # Enhanced gradient descent with better convergence control
    def optimize_with_gradient_descent(points, max_iter=400, tolerance=1e-5):
        """Optimize point configuration using gradient descent with momentum"""
        current_points = points.copy()
        velocity = np.zeros_like(current_points)
        momentum = 0.95
        learning_rate = 0.02
        
        prev_energy = float('inf')
        patience_counter = 0
        patience_limit = 30
        
        for iteration in range(max_iter):
            # Compute energy and gradient
            current_energy = energy_model(current_points)
            
            # Check for convergence
            if abs(prev_energy - current_energy) < tolerance:
                patience_counter += 1
                if patience_counter > patience_limit:
                    break
            else:
                patience_counter = 0
                
            prev_energy = current_energy
            
            # Compute gradient
            grad = gradient_energy(current_points)
            
            # Apply momentum and update
            velocity = momentum * velocity - learning_rate * grad
            current_points += velocity
            
            # Project back to valid bounds [0,1]
            current_points = np.clip(current_points, 0, 1)
            
            # Occasionally reproject to avoid drift
            if iteration % 50 == 0:
                current_points = np.clip(current_points, 0, 1)
                
        return current_points
    
    # Multi-start optimization with weighted selection based on initial quality
    def multi_start_optimization():
        """Run multiple optimizations from different starting points with intelligent selection"""
        best_points = None
        best_ratio = 0
        
        # Generate multiple initial configurations
        initials = generate_initial_configurations()
        
        # Track quality of each initialization
        initial_ratios = []
        for init_name, initial_points in initials:
            ratio = compute_min_max_ratio(initial_points)
            initial_ratios.append((init_name, initial_points, ratio))
            
        # Sort by initial quality (descending)
        initial_ratios.sort(key=lambda x: x[2], reverse=True)
        
        # Try the best initial configurations first
        for init_name, initial_points, initial_ratio in initial_ratios[:5]:  # Try top 5
            try:
                # Optimize using gradient descent
                optimized_points = optimize_with_gradient_descent(
                    initial_points, 
                    max_iter=400,  # More iterations for better convergence
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
            for _, initial_points, ratio in initial_ratios:
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
        iterations_per_temp = 100  # More iterations per temperature step for better exploration
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Annealing loop with better stopping conditions
        while temperature > min_temperature:
            for _ in range(iterations_per_temp):
                # Make small random perturbations
                new_points = current_points.copy()
                idx = np.random.randint(0, 16)
                # Use smaller perturbation for fine-tuning
                new_points[idx] += np.random.normal(0, 0.003, 2)
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
    
    # Add L-BFGS-B optimization for final high-precision refinement (like INSPIRATION 1)
    def lbfgsb_optimization(initial_points):
        """Use L-BFGS-B optimization for high precision refinement"""
        def objective(x_flat):
            # Reshape flat array back to points
            points = x_flat.reshape(-1, 2)
            
            # Compute pairwise distances
            distances = pdist(points)
            
            if len(distances) == 0:
                return float('inf')
                
            # Calculate min and max distances
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            # Avoid division by zero - return large penalty
            if d_max <= 0:
                return float('inf')
                
            # Return negative ratio (since we want to maximize ratio)
            return -d_min / d_max
        
        try:
            x0 = initial_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                return optimized_points
        except Exception:
            pass
        return initial_points
    
    # Main optimization pipeline
    try:
        # Run multi-start optimization to find good starting points
        points = multi_start_optimization()
        
        # First refinement with simulated annealing
        refined_points = simulated_annealing_refinement(points)
        
        # Then apply high-precision L-BFGS-B optimization
        final_points = lbfgsb_optimization(refined_points)
        
        # Return the better of the two (final optimization or refined SA)
        original_ratio = compute_min_max_ratio(refined_points)
        final_ratio = compute_min_max_ratio(final_points)
        
        return final_points if final_ratio > original_ratio else refined_points
        
    except Exception as e:
        # Fallback to golden spiral approach with better parameters
        n = 16
        points = np.zeros((n, 2))
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(n):
            angle = 2 * np.pi * i / phi
            # Use more even radial distribution
            radius = 0.4 * np.sqrt(i / (n - 1)) if i < n - 1 else 0.4
            points[i] = [
                0.5 + radius * np.cos(angle),
                0.5 + radius * np.sin(angle)
            ]
        
        np.random.seed(42)
        points += np.random.normal(0, 0.012, points.shape)
        points = np.clip(points, 0, 1)
        return points


# EVOLVE-BLOCK-END
