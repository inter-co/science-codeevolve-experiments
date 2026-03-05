# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0
    
    # Compute pairwise distances
    distances = pdist(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist == 0:
        return 0
    
    return min_dist / max_dist


def constraint_bounds(x_flat):
    """Constraint function ensuring all points stay within [0,1] x [0,1]"""
    points = x_flat.reshape(-1, 2)
    # Ensure all points are within [0,1] x [0,1]
    return np.concatenate([
        points[:, 0],           # x-coordinates >= 0
        1 - points[:, 0],       # x-coordinates <= 1
        points[:, 1],           # y-coordinates >= 0
        1 - points[:, 1]        # y-coordinates <= 1
    ])


def objective(x_flat):
    """Objective function to minimize (negative of ratio to maximize ratio)"""
    points = x_flat.reshape(-1, 2)
    
    # Compute pairwise distances
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0
        
    # Calculate min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    # Avoid division by zero
    if d_max <= 0:
        return 0
        
    # Return negative ratio (since we want to maximize ratio)
    return -d_min / d_max


def energy_minimization_initialization(n_points=16, num_attempts=5):
    """Initialize points using energy minimization approach"""
    best_energy = float('inf')
    best_points = None
    
    for attempt in range(num_attempts):
        # Start with random initialization
        np.random.seed(1000 + attempt)
        points = np.random.rand(n_points, 2)
        
        # Simple energy minimization using repulsion force with boundary constraints
        for _ in range(2000):  # More iterations for better convergence
            # Compute forces between all pairs
            forces = np.zeros_like(points)
            for i in range(n_points):
                for j in range(i+1, n_points):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq > 1e-10:  # Avoid division by zero
                        force_magnitude = 1.0 / (dist_sq**1.5)  # Repulsive force
                        forces[i] += force_magnitude * diff
                        forces[j] -= force_magnitude * diff
            
            # Apply forces with damping
            points += 0.005 * forces
            
            # Keep within bounds (reflective boundaries to maintain constraints)
            points = np.clip(points, 0, 1)
        
        # Evaluate energy (sum of inverse squared distances)
        distances = pdist(points)
        if len(distances) > 0:
            energy = np.sum(1.0 / (distances**2))
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
    
    return best_points if best_points is not None else np.random.rand(n_points, 2)


def golden_spiral_initialization():
    """Create initial configuration using golden spiral pattern"""
    n = 16
    points = np.zeros((n, 2))
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # Generate points using golden spiral with slight perturbations
    for i in range(n):
        angle = 2 * np.pi * i / phi
        radius = np.sqrt(i / (n - 1)) if i < n - 1 else 1.0
        points[i] = [
            0.5 + 0.4 * radius * np.cos(angle),
            0.5 + 0.4 * radius * np.sin(angle)
        ]
    
    # Add structured perturbations to improve initial spread
    np.random.seed(42)
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, 0, 1)
    
    return points


def hexagonal_lattice_initialization():
    """Create initial configuration using hexagonal lattice pattern"""
    points = np.zeros((16, 2))
    sqrt3 = np.sqrt(3)
    row_height = sqrt3 / 2
    col_spacing = 1.0
    row_spacing = row_height
    
    idx = 0
    for i in range(4):
        for j in range(4):
            if idx >= 16:
                break
            x = j * col_spacing + (i % 2) * col_spacing * 0.5
            y = i * row_spacing
            points[idx, 0] = x / (3.5 * col_spacing)
            points[idx, 1] = y / (3.5 * row_spacing)
            idx += 1
    
    # Add small random perturbations to break symmetry
    np.random.seed(999)
    points += np.random.normal(0, 0.015, (16, 2))
    points = np.clip(points, 0, 1)
    return points


def generate_multiple_initializations():
    """Generate several diverse initial configurations inspired by multiple strategies"""
    configs = []
    
    # Golden spiral (most effective approach)
    configs.append(("golden", golden_spiral_initialization()))
    
    # Hexagonal lattice pattern
    configs.append(("hexagonal", hexagonal_lattice_initialization()))
    
    # Energy-based initialization
    configs.append(("energy", energy_minimization_initialization()))
    
    # Grid pattern with better distribution
    grid_points = np.zeros((16, 2))
    for i in range(4):
        for j in range(4):
            grid_points[i*4 + j] = [i/3.0 + 0.125, j/3.0 + 0.125]
    np.random.seed(123)
    grid_points += np.random.normal(0, 0.01, grid_points.shape)
    grid_points = np.clip(grid_points, 0, 1)
    configs.append(("grid", grid_points))
    
    # Random with different seed
    np.random.seed(456)
    random_points = np.random.uniform(0, 1, (16, 2))
    configs.append(("random", random_points))
    
    # Another golden spiral with different seed
    np.random.seed(789)
    golden2 = golden_spiral_initialization()
    configs.append(("golden2", golden2))
    
    # Concentrated pattern near center
    np.random.seed(222)
    centered_points = np.random.uniform(0.2, 0.8, (16, 2))
    configs.append(("centered", centered_points.copy()))
    
    return configs


def hybrid_optimization(initial_points):
    """Use hybrid optimization approach combining local and global methods"""
    best_points = None
    best_ratio = 0
    
    # Strategy 1: Local optimization with SLSQP
    try:
        x0 = initial_points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        cons = {'type': 'ineq', 'fun': constraint_bounds}
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
    except Exception:
        pass
    
    # Strategy 2: Global optimization with differential evolution if local failed
    if best_points is None or best_ratio < 0.05:
        try:
            bounds = [(0, 1) for _ in range(32)]
            result = differential_evolution(
                objective,
                bounds,
                maxiter=200,
                popsize=20,
                seed=42,
                disp=False,
                tol=1e-12
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            pass
    
    # Strategy 3: Refined local optimization if we have a good starting point
    if best_points is not None:
        try:
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            cons = {'type': 'ineq', 'fun': constraint_bounds}
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            pass
    
    return best_points, best_ratio


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction, energy-based initialization,
    and both local and global optimization methods for robust global search.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initializations()
    
    best_points = None
    best_ratio = 0
    
    # Try optimization from each initial configuration using hybrid approach
    for config_name, initial_points in initial_configs:
        try:
            optimized_points, ratio = hybrid_optimization(initial_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # If no optimization worked, return the best of the initial configurations
    if best_points is None:
        # Evaluate all initial configurations and return the best
        for config_name, initial_points in initial_configs:
            ratio = compute_min_max_ratio(initial_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = initial_points.copy()
    
    # Final refinement with high precision optimization on the best result
    if best_points is not None:
        final_points, final_ratio = hybrid_optimization(best_points)
        if final_ratio > best_ratio:
            return final_points
    
    return best_points if best_points is not None else golden_spiral_initialization()


# EVOLVE-BLOCK-END
