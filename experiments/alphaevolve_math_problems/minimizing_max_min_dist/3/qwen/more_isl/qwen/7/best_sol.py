# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and global optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    
    # Strategy 1: Try multiple geometric initializations
    initial_configs = []
    
    # Config 1: Icosahedral + Fibonacci points (existing approach)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices[0])
    
    points1 = vertices.copy()
    for i in range(2):
        idx = 12 + i
        y = -1 + (2 * idx / 13)
        radius = np.sqrt(1 - y * y)
        golden_angle = 2 * np.pi * (3 - np.sqrt(5))  
        angle = idx * golden_angle
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        points1 = np.vstack([points1, [x, y, z]])
    
    # Normalize
    norms = np.linalg.norm(points1, axis=1)
    points1 = points1 / norms[:, np.newaxis]
    initial_configs.append(('icosahedral_fibonacci', points1))
    
    # Config 2: Octahedral + 2 points (different geometry)
    octahedron_points = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    
    # Add 8 more points using a more uniform distribution
    points2 = octahedron_points.copy()
    # Add 8 more points in a way that maintains good spread
    for i in range(8):
        # Distribute points around the sphere more evenly
        theta = np.pi * (3 - np.sqrt(5)) * i
        y = 1 - 2 * i / 9  # y from 1 to -1
        radius = np.sqrt(1 - y * y)
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        points2 = np.vstack([points2, [x, y, z]])
    
    # Normalize
    norms = np.linalg.norm(points2, axis=1)
    points2 = points2 / norms[:, np.newaxis]
    initial_configs.append(('octahedral', points2))
    
    # Config 3: Random but constrained initialization for diversity
    points3 = np.random.rand(14, 3) * 2 - 1  # Range [-1, 1]
    norms = np.linalg.norm(points3, axis=1)
    points3 = points3 / norms[:, np.newaxis]
    initial_configs.append(('random_constrained', points3))
    
    # NEW APPROACH 2: Algebraic number theory approach using cyclotomic fields
    # Using 14th roots of unity projected onto 3D space
    def generate_cyclotomic_points():
        # Generate points based on 14th roots of unity in complex plane
        # Then project to 3D using algebraic relationships
        n = 14
        angles = [2 * np.pi * k / n for k in range(n)]
        
        # Use the fact that we can construct points from roots of unity
        # For 14 points, we'll use a combination of symmetries and algebraic properties
        points = []
        
        # First, create points using fundamental cyclotomic relations
        for k in range(14):
            angle = 2 * np.pi * k / 14
            # Use algebraic properties: cos(2πk/14) and sin(2πk/14) 
            # But we'll use a more structured approach based on 7-gon symmetry
            # since 14 = 2 × 7
            
            # Generate points based on 7-gon vertices with symmetry considerations
            # This creates a structure with high symmetry and good distribution
            theta = angle
            phi = 2 * np.pi * (k // 2) / 7  # Additional angular parameter
            
            # Convert to 3D using spherical coordinates
            # Use a modified approach that leverages the algebraic structure
            x = np.cos(phi) * np.cos(theta)
            y = np.cos(phi) * np.sin(theta)
            z = np.sin(phi)
            
            points.append([x, y, z])
            
        # Normalize all points to unit sphere
        points = np.array(points)
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        return points
    
    # Add cyclotomic approach as a fourth initial configuration
    points4 = generate_cyclotomic_points()
    initial_configs.append(('cyclotomic', points4))
    
    # Convert to a proper optimization setup
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return -np.inf
            
        # Minimize negative of ratio (maximize ratio)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return -np.inf
            
        ratio = min_dist / max_dist
        return -ratio  # Negative because we want to maximize
    
    # Enhanced optimization strategy with better parameter tuning
    bounds = [(-1, 1) for _ in range(42)]  # 14 points * 3 coordinates
    
    best_ratio = -np.inf
    best_points = None
    
    # Try optimization from each initial configuration
    for config_name, initial_points in initial_configs:
        # Strategy 1: Differential Evolution with more aggressive parameters
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=50,      # Increased iterations
                popsize=12,      # Increased population size
                mutation=(0.8, 1),  # Slightly higher mutation rate
                recombination=0.9,  # Higher recombination rate
                seed=42,
                disp=False
            )
            
            if de_result.success:
                refined_points = de_result.x.reshape(-1, 3)
                norms = np.linalg.norm(refined_points, axis=1)
                refined_points = refined_points / norms[:, np.newaxis]
                
                # Local refinement with more aggressive settings
                x0 = refined_points.flatten()
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    norms = np.linalg.norm(final_points, axis=1)
                    final_points = final_points / norms[:, np.newaxis]
                    
                    # Evaluate final solution
                    distances = pdist(final_points)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()
        except Exception:
            pass
    
    # Strategy 2: Multiple restarts with diverse perturbations
    methods = ['L-BFGS-B', 'SLSQP', 'TNC']
    
    # Try multiple restarts with different strategies
    for restart in range(6):  # More restarts
        np.random.seed(1000 + restart)
        
        # Choose a random initial configuration for diversity
        initial_idx = np.random.randint(len(initial_configs))
        points = initial_configs[initial_idx][1].copy()
        
        # Apply different types of perturbations based on restart number
        if restart < 2:
            # Small perturbation
            perturbed_points = points + np.random.normal(0, 0.02, points.shape)
        elif restart < 4:
            # Medium perturbation
            perturbed_points = points + np.random.normal(0, 0.04, points.shape)
        else:
            # Larger perturbation with some systematic adjustment
            perturbed_points = points + np.random.normal(0, 0.06, points.shape)
            # Adjust for better spread
            for i in range(14):
                norms = np.linalg.norm(perturbed_points[i])
                if norms > 0:
                    perturbed_points[i] = perturbed_points[i] / norms
        
        perturbed_norms = np.linalg.norm(perturbed_points, axis=1)
        perturbed_points = perturbed_points / perturbed_norms[:, np.newaxis]
        
        x0 = perturbed_points.flatten()
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    options={'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    distances = pdist(final_points)
                    
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()
                                
            except Exception:
                continue
    
    # If no optimization worked, return the best initial configuration
    if best_points is None:
        # Return the first configuration as fallback
        return initial_configs[0][1]
    
    return best_points


# EVOLVE-BLOCK-END
