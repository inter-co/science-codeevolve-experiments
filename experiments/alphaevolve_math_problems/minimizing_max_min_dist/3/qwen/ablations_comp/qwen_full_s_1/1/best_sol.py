# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import random
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, multi-start optimization, 
    and advanced physics-based refinement to exceed the benchmark of 0.4898.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio"""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0 or np.max(distances) == 0:
            return 0
        return np.min(distances) / np.max(distances)
    
    def project_to_sphere(points):
        """Project points onto unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms < 1e-10, 1.0, norms)
        return points / norms
    
    def generate_icosahedral_initialization():
        """Generate initial points using icosahedral symmetry."""
        # Vertices of regular icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        
        # Normalize to unit sphere
        normalized_vertices = []
        for x, y, z in vertices:
            norm = np.sqrt(x*x + y*y + z*z)
            normalized_vertices.append([x/norm, y/norm, z/norm])
        
        # Return first 12 vertices plus 2 more points near poles
        points = normalized_vertices[:12]
        points.append([0, 0, 0.95])  # Near north pole
        points.append([0, 0, -0.95])  # Near south pole
        
        return np.array(points)
    
    def generate_cube_sphere_initialization():
        """Generate points on a cube inscribed in the sphere."""
        # Cube vertices (normalized)
        cube_vertices = [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]
        
        # Normalize to unit sphere
        normalized_cube = []
        for x, y, z in cube_vertices:
            norm = np.sqrt(x*x + y*y + z*z)
            normalized_cube.append([x/norm, y/norm, z/norm])
        
        # Add 6 more points at face centers
        face_centers = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ]
        
        # Normalize face centers
        for x, y, z in face_centers:
            norm = np.sqrt(x*x + y*y + z*z)
            normalized_cube.append([x/norm, y/norm, z/norm])
        
        # Return first 14 points (cube vertices + face centers)
        return np.array(normalized_cube[:14])
    
    def generate_fibonacci_sphere_initialization(n=14):
        """Generate points using Fibonacci spiral on sphere."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def generate_algebraic_field_points():
        """
        Generate points using algebraic number field constructions.
        This approach uses specific algebraic relationships to create well-distributed points.
        """
        # Use the roots of unity and related algebraic numbers
        # Create a configuration inspired by the E8 lattice projection
        # We'll construct points using combinations of sqrt(2) and other algebraic numbers
        
        # Generate points with coordinates involving sqrt(2) and sqrt(3)
        sqrt2 = np.sqrt(2)
        sqrt3 = np.sqrt(3)
        
        # These points are constructed to have specific algebraic relationships
        points = []
        
        # Add vertices of a regular octahedron with scaling factors
        octahedron_vertices = [
            [sqrt2, 0, 0], [-sqrt2, 0, 0],
            [0, sqrt2, 0], [0, -sqrt2, 0],
            [0, 0, sqrt2], [0, 0, -sqrt2]
        ]
        
        points.extend(octahedron_vertices)
        
        # Add 8 more points to make 14 total
        # Using combinations of 1/sqrt(2) and 1/sqrt(3) 
        additional_points = [
            [1/sqrt2, 1/sqrt2, 0],
            [1/sqrt2, -1/sqrt2, 0],
            [-1/sqrt2, 1/sqrt2, 0],
            [-1/sqrt2, -1/sqrt2, 0],
            [1/sqrt3, 1/sqrt3, 1/sqrt3],
            [1/sqrt3, 1/sqrt3, -1/sqrt3],
            [1/sqrt3, -1/sqrt3, 1/sqrt3],
            [-1/sqrt3, 1/sqrt3, 1/sqrt3]
        ]
        
        points.extend(additional_points[:8])
        
        points_array = np.array(points[:14])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points_array, axis=1, keepdims=True)
        points_array = points_array / norms
        
        return points_array
    
    def compute_repulsive_forces(points):
        """Compute repulsive forces between all pairs of points"""
        n = len(points)
        forces = np.zeros_like(points)
        
        # Vectorized computation for efficiency
        for i in range(n):
            diff = points[i] - points  # All differences at once
            diff[i] = 0  # Remove self-difference
            dist_sq = np.sum(diff**2, axis=1)
            
            # Avoid division by zero and very small distances
            mask = dist_sq > 1e-12
            if np.any(mask):
                force_magnitudes = 1.0 / (dist_sq[mask] * np.sqrt(dist_sq[mask]))
                forces[i] = np.sum(force_magnitudes[:, np.newaxis] * diff[mask], axis=0)
                
        return forces
    
    def physics_refinement(initial_points, max_iter=200, learning_rate=0.005):
        """Refine points using physics-based repulsion simulation with early stopping"""
        points = initial_points.copy()
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(points)
        
        # Track improvement for early stopping
        last_improvement = 0
        improvement_threshold = 1e-12
        
        for iteration in range(max_iter):
            # Compute forces (vectorized version)
            forces = compute_repulsive_forces(points)
            
            # Update positions with small step size
            points += learning_rate * forces
            
            # Project back to sphere
            points = project_to_sphere(points)
            
            # Check for improvement
            current_ratio = compute_min_max_ratio(points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()
                last_improvement = iteration
            
            # Early stopping based on improvement
            if iteration - last_improvement > 50 and abs(current_ratio - best_ratio) < improvement_threshold:
                break
                
        return best_points
    
    def simulated_annealing():
        """Run simulated annealing optimization with enhanced parameters."""
        best_points = None
        best_ratio = -np.inf
        
        # Multiple random starts with diverse initializations
        for start_iter in range(30):  # Increased from 25 to 30 for better exploration
            # Initialize with different strategies
            if start_iter % 5 == 0:
                points = generate_icosahedral_initialization()
            elif start_iter % 5 == 1:
                points = generate_cube_sphere_initialization()
            elif start_iter % 5 == 2:
                points = generate_fibonacci_sphere_initialization(14)
            elif start_iter % 5 == 3:
                points = generate_algebraic_field_points()
            else:
                # Random initialization as fallback
                points = np.random.randn(14, 3)
                points = project_to_sphere(points)
            
            # Add some random noise to break symmetry
            np.random.seed(start_iter)
            points += np.random.normal(0, 0.05, points.shape)
            points = project_to_sphere(points)
            
            # Convert to flat array for optimization
            points_flat = points.flatten()
            
            # Optimization parameters - tuned for faster convergence
            temp = 1.0
            cooling_rate = 0.995
            min_temp = 1e-6
            iterations_per_temp = 150  # Reduced for time efficiency
            
            current_points_flat = points_flat.copy()
            current_ratio = compute_min_max_ratio(current_points_flat.reshape(-1, 3))
            
            # Store best solution found
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points_flat.copy()
            
            # Simulated annealing loop
            iteration_count = 0
            max_iterations = 3000  # Reduced iterations to stay within time limits
            while temp > min_temp and iteration_count < max_iterations:
                for _ in range(iterations_per_temp):
                    # Create neighbor by perturbing one point
                    new_points_flat = current_points_flat.copy()
                    point_idx = random.randint(0, 13)
                    dim_idx = random.randint(0, 2)
                    
                    # Perturb the point with adaptive step size
                    step_size = 0.03 if temp > 0.1 else 0.008
                    new_points_flat[point_idx * 3 + dim_idx] += np.random.normal(0, step_size)
                    
                    # Project back to sphere
                    new_points = new_points_flat.reshape(-1, 3)
                    new_points = project_to_sphere(new_points)
                    new_points_flat = new_points.flatten()
                    
                    # Evaluate new solution
                    new_ratio = compute_min_max_ratio(new_points)
                    
                    # Accept or reject based on Metropolis criterion
                    if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
                        current_points_flat = new_points_flat.copy()
                        current_ratio = new_ratio
                        
                        if current_ratio > best_ratio:
                            best_ratio = current_ratio
                            best_points = current_points_flat.copy()
                
                temp *= cooling_rate
                iteration_count += 1
            
            # Early termination if we're getting very good results
            if best_ratio > 0.45:
                break
        
        return best_points.reshape(-1, 3) if best_points is not None else generate_fibonacci_sphere_initialization(14)
    
    # Use simulated annealing which proved to be more effective than multi-start in this case
    final_points = simulated_annealing()
    
    # Final refinement with physics-based approach
    final_points = physics_refinement(final_points, max_iter=150, learning_rate=0.003)
    
    # Final optimization with scipy optimizer for high precision
    points_flat = final_points.flatten()
    
    # Use L-BFGS-B for fine-tuning with tighter tolerances
    def objective_function(x_flat):
        points = x_flat.reshape(-1, 3)
        points = project_to_sphere(points)
        ratio = compute_min_max_ratio(points)
        return -ratio if ratio > 0 else 1e10  # Large penalty for invalid cases
    
    try:
        result = minimize(
            objective_function,
            points_flat,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            final_points = project_to_sphere(final_points)
    except Exception:
        # If optimization fails, use the best we have
        pass
    
    return final_points


# EVOLVE-BLOCK-END
