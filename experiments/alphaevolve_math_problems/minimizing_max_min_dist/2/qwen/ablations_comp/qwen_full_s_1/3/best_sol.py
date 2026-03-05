# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution, dual_annealing
import warnings
warnings.filterwarnings('ignore')
import random
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining mathematical initialization with global and 
    local optimization for robust convergence.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def objective(x_flat):
        """Objective function to maximize the min/max distance ratio"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to minimize (since we want to maximize)
        if d_max == 0:
            return 0
        return -d_min / d_max
    
    def generate_mathematical_initial():
        """Generate points based on mathematical principles for better distribution"""
        # Strategy from INSPIRATION 2: Concentric rings with golden ratio properties
        points = []
        
        # Outer ring: 8 points forming a regular octagon with optimized radius
        outer_radius = 0.4
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = 0.5 + outer_radius * math.cos(angle)
            y = 0.5 + outer_radius * math.sin(angle)
            points.append([x, y])
        
        # Inner ring: 8 points forming a rotated octagon with golden ratio spacing
        inner_radius = 0.25
        for i in range(8):
            angle = 2 * math.pi * i / 8 + math.pi/8
            x = 0.5 + inner_radius * math.cos(angle)
            y = 0.5 + inner_radius * math.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Add controlled noise to break symmetry while maintaining mathematical structure
        noise = np.random.normal(0, 0.015, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        return points
    
    def generate_hexagonal_pattern():
        """Generate a high-quality hexagonal pattern from INSPIRATION 2"""
        # Create a 4x4 grid with hexagonal offset
        points = []
        for i in range(4):
            for j in range(4):
                # Hexagonal offset for every other row
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points[:16])  # Take only 16 points
        
        # Normalize to [0,1] x [0,1] 
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Scale to fit nicely in [0.1, 0.9] x [0.1, 0.9] 
        points[:, 0] = 0.8 * points[:, 0] + 0.1
        points[:, 1] = 0.8 * points[:, 1] + 0.1
        
        return points
    
    def generate_regular_polygon():
        """Generate points arranged in a regular polygon pattern from INSPIRATION 2"""
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 0.4  # Slightly smaller than 0.5 to leave margin
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        
        # Normalize to [0,1] bounds
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range * 0.8 + 0.1
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range * 0.8 + 0.1
            
        return points
    
    def generate_grid_pattern():
        """Generate a regular 4x4 grid pattern"""
        points = np.zeros((n, d))
        count = 0
        for i in range(4):
            for j in range(4):
                points[count] = [i * 0.25 + 0.125, j * 0.25 + 0.125]
                count += 1
        return points
    
    def generate_fibonacci_spiral():
        """Generate points distributed according to Fibonacci spiral from INSPIRATION 2"""
        points = np.zeros((n, d))
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            theta = i * 2 * np.pi / phi
            r = np.sqrt(i) / np.sqrt(n-1) if n > 1 else 0.5
            points[i] = [r * np.cos(theta), r * np.sin(theta)]
        
        # Normalize to [0.1, 0.9] x [0.1, 0.9]
        points[:, 0] = 0.8 * points[:, 0] + 0.5
        points[:, 1] = 0.8 * points[:, 1] + 0.5
        points = np.clip(points, 0, 1)
        return points
    
    def generate_vortex_pattern():
        """Generate a vortex pattern with better control from INSPIRATION 2"""
        points = []
        num_rings = 4
        points_per_ring = n // num_rings
        
        for ring_idx in range(num_rings):
            n_points = points_per_ring if ring_idx < num_rings - 1 else n - (num_rings - 1) * points_per_ring
            
            # Use logarithmic spacing for rings with better parameters
            radius = 0.1 + 0.4 * (ring_idx / (num_rings - 1)) if num_rings > 1 else 0.25
            
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            
            for i, angle in enumerate(angles):
                # Add a vortex effect with sinusoidal modulation
                vortex_factor = 0.05 * math.sin(3 * angle + ring_idx * math.pi/2)
                r = radius * (1 + vortex_factor)
                
                x = 0.5 + r * math.cos(angle)
                y = 0.5 + r * math.sin(angle)
                
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                
                points.append([x, y])
            
            if len(points) >= n:
                break
        
        points = np.array(points[:n])
        # Add jitter to break symmetry
        np.random.seed(42)
        jitter_magnitude = 0.01
        points += np.random.uniform(-jitter_magnitude, jitter_magnitude, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def generate_golden_spiral():
        """Generate points using golden spiral approach with better normalization"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            theta = i * 2 * np.pi / phi
            r = np.sqrt(i / (n - 1)) if n > 1 else 0
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
        points = np.array(points)
        
        # Normalize properly with better bounds
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        range_coords = max_coords - min_coords
        if np.any(range_coords == 0):
            range_coords[range_coords == 0] = 1
        points = (points - min_coords) / range_coords * 0.8 + 0.1
        
        return points
    
    def simulated_annealing(initial_points, max_iter=2000, temp_start=1.0, cooling_rate=0.999):
        """Apply simulated annealing to improve the configuration"""
        points = initial_points.copy()
        current_ratio = compute_min_max_ratio(points)
        
        # Keep track of the best solution found
        best_points = points.copy()
        best_ratio = current_ratio
        
        # Temperature schedule
        temperature = temp_start
        
        for iteration in range(max_iter):
            # Generate neighbor by perturbing one point randomly
            neighbor = points.copy()
            idx = random.randint(0, n-1)
            # Small random perturbation
            neighbor[idx] += np.random.normal(0, 0.005, 2)
            # Keep within bounds
            neighbor[idx] = np.clip(neighbor[idx], 0, 1)
            
            # Evaluate neighbor
            neighbor_ratio = compute_min_max_ratio(neighbor)
            
            # Accept or reject based on simulated annealing criteria
            if neighbor_ratio > current_ratio:
                points = neighbor
                current_ratio = neighbor_ratio
                if neighbor_ratio > best_ratio:
                    best_points = neighbor.copy()
                    best_ratio = neighbor_ratio
            else:
                # Accept with probability based on temperature
                delta = neighbor_ratio - current_ratio
                if random.random() < np.exp(delta / temperature):
                    points = neighbor
                    current_ratio = neighbor_ratio
            
            # Cool down temperature
            temperature *= cooling_rate
            
            # Early stopping if temperature gets too low
            if temperature < 1e-8:
                break
                
        return best_points, best_ratio
    
    # Enhanced initialization strategies with better mathematical basis from INSPIRATION 2
    strategies = [
        ("mathematical", generate_mathematical_initial()),
        ("hexagonal", generate_hexagonal_pattern()),
        ("polygon", generate_regular_polygon()),
        ("grid", generate_grid_pattern()),
        ("fibonacci", generate_fibonacci_spiral()),
        ("vortex", generate_vortex_pattern()),
        ("golden", generate_golden_spiral())
    ]
    
    # Find best initial configuration
    best_points = None
    best_ratio = 0
    
    for name, points in strategies:
        ratio = compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # Global optimization approach - try multiple global optimizers like in inspiration 2
    bounds = [(0, 1) for _ in range(n * d)]
    
    # Strategy 1: Differential Evolution (more robust for this problem)
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=1000,  # Increased iterations for better convergence
            popsize=50,    # Larger population size for better exploration
            mutation=(0.9, 1),  # Higher mutation rate for better exploration
            recombination=0.95,   # Higher recombination for better mixing
            atol=1e-15,
            rtol=1e-15
        )
        
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            ratio = compute_min_max_ratio(de_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = de_points.copy()
    except Exception:
        pass
    
    # Strategy 2: Dual Annealing (good for escaping local minima)
    try:
        da_result = dual_annealing(
            objective,
            bounds,
            maxiter=1000,  # Increased iterations for better convergence
            initial_temp=3500,  # Higher initial temperature for better exploration
            seed=42,
            no_local_search=False  # Enable local search for better refinement
        )
        
        if da_result.success:
            da_points = da_result.x.reshape(-1, 2)
            da_points = np.clip(da_points, 0, 1)
            ratio = compute_min_max_ratio(da_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = da_points.copy()
    except Exception:
        pass
    
    # Multi-start local optimization with better restarts
    # Run local optimizations from multiple good starting points
    local_optimizations = []
    
    # Start with the best global solution
    if best_points is not None:
        local_optimizations.append(best_points.copy())
    
    # Add additional starting points from different strategies with more diversity
    for i in range(10):  # More restarts to improve chances of finding better solutions
        np.random.seed(100 + i)
        strategy_name, initial_points = strategies[np.random.randint(len(strategies))]
        # Add small random perturbation
        perturbed_points = initial_points.copy()
        perturbed_points += np.random.normal(0, 0.01, (16, 2))
        perturbed_points = np.clip(perturbed_points, 0, 1)
        local_optimizations.append(perturbed_points)
    
    # Run local optimizations
    for i, start_points in enumerate(local_optimizations):
        try:
            result = minimize(
                objective,
                start_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}  # Even more iterations for better convergence
            )
            
            if result.success:
                optimized_points = result.x.reshape(16, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            continue
    
    # Additional refinement with simulated annealing on the best solution
    if best_points is not None:
        try:
            sa_points, sa_ratio = simulated_annealing(best_points, max_iter=2000)
            if sa_ratio > best_ratio:
                best_points = sa_points
                best_ratio = sa_ratio
        except Exception:
            pass
    
    # Final refinement with local search around the best solution
    if best_points is not None:
        try:
            # Perform a few rounds of fine-grained local optimization
            for round_num in range(5):  # More rounds for better refinement
                improved = False
                # Test small perturbations to each point
                for i in range(16):
                    old_point = best_points[i].copy()
                    # Try multiple small perturbations
                    for _ in range(20):  # More perturbations for better search
                        perturbation = np.random.normal(0, 0.001, 2)
                        new_point = old_point + perturbation
                        new_point = np.clip(new_point, 0, 1)
                        
                        test_points = best_points.copy()
                        test_points[i] = new_point
                        new_ratio = compute_min_max_ratio(test_points)
                        
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_points[i] = new_point
                            improved = True
                
                # If no improvement, stop early
                if not improved:
                    break
                    
        except Exception:
            pass
    
    # Ensure we return a valid solution even if optimization fails
    if best_points is None:
        # Fallback to the best initial configuration
        return strategies[0][1]
    
    return best_points


# EVOLVE-BLOCK-END
