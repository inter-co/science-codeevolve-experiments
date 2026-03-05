# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import cKDTree
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a multi-scale hierarchical optimization approach with adaptive strategies.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all pairs"""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0
    
    def compute_pairwise_distances(points):
        """Compute full pairwise distance matrix"""
        return squareform(pdist(points))
    
    def create_initial_configurations():
        """Create multiple diverse initial configurations"""
        configs = {}
        
        # Configuration 1: Regular 16-gon with slight perturbations
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        points = []
        for angle in angles:
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        configs['regular_polygon'] = np.array(points)
        
        # Configuration 2: Concentric rings
        points = []
        # Outer ring: 12 points
        for i in range(12):
            angle = 2 * np.pi * i / 12
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        # Inner ring: 4 points
        for i in range(4):
            angle = 2 * np.pi * i / 4
            x = 0.5 + 0.15 * np.cos(angle)
            y = 0.5 + 0.15 * np.sin(angle)
            points.append([x, y])
        configs['concentric_rings'] = np.array(points)
        
        # Configuration 3: Hexagonal grid with perturbations
        points = []
        for i in range(4):
            for j in range(4):
                x = j + 0.5 * (i % 2)
                y = i * math.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points[:16])
        # Normalize to [0,1]
        if np.max(points[:, 0]) > np.min(points[:, 0]):
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / (np.max(points[:, 0]) - np.min(points[:, 0]))
        if np.max(points[:, 1]) > np.min(points[:, 1]):
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / (np.max(points[:, 1]) - np.min(points[:, 1]))
        
        # Add perturbations
        np.random.seed(42)
        points += np.random.normal(0, 0.02, points.shape)
        configs['hexagonal_grid'] = np.clip(points, 0, 1)
        
        # Configuration 4: Fibonacci-like distribution
        points = []
        phi = math.pi * (3 - math.sqrt(5))  # golden angle
        for i in range(16):
            y = 1 - (i / float(16 - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append([(x + 1) / 2, (z + 1) / 2])
        configs['fibonacci'] = np.array(points)
        
        return configs
    
    def hierarchical_clustering_optimization(initial_points, max_iter=300):
        """Optimize using hierarchical clustering to identify natural groups"""
        points = initial_points.copy()
        n_points = len(points)
        
        # Create a multi-stage optimization approach
        for stage in range(3):
            # Stage 1: Coarse optimization (larger steps)
            if stage == 0:
                step_size = 0.05
                iter_count = 100
            elif stage == 1:
                step_size = 0.02
                iter_count = 100
            else:  # stage == 2
                step_size = 0.005
                iter_count = 100
            
            # Perform iterative optimization with adaptive step sizes
            for iteration in range(iter_count):
                # Calculate current state
                current_ratio = compute_min_max_ratio(points)
                
                # For each point, compute the effect of small perturbations
                for i in range(n_points):
                    # Try small perturbations in different directions
                    best_delta = None
                    best_ratio = current_ratio
                    
                    # Try several perturbation directions
                    for _ in range(10):
                        # Generate random perturbation
                        delta = np.random.normal(0, step_size, 2)
                        
                        # Test this perturbation
                        test_points = points.copy()
                        test_points[i] += delta
                        
                        # Keep within bounds
                        test_points = np.clip(test_points, 0, 1)
                        
                        # Check if this improves the ratio
                        test_ratio = compute_min_max_ratio(test_points)
                        
                        if test_ratio > best_ratio:
                            best_ratio = test_ratio
                            best_delta = delta
                    
                    # Apply the best improvement if any
                    if best_delta is not None:
                        points[i] += best_delta
                
                # Occasionally perform global adjustment based on clustering
                if iteration % 50 == 0 and stage > 0:
                    # Cluster points to understand local structure
                    tree = cKDTree(points)
                    # Find neighbors within a certain radius
                    neighbors = tree.query_ball_tree(tree, r=0.1)
                    if len(neighbors) > 0:
                        # Adjust points to reduce local clustering
                        for i in range(n_points):
                            if len(neighbors[i]) > 3:  # If point has many close neighbors
                                # Repel from cluster center
                                neighbor_points = points[neighbors[i]]
                                center = np.mean(neighbor_points, axis=0)
                                direction = points[i] - center
                                if np.linalg.norm(direction) > 1e-10:
                                    points[i] += 0.001 * direction / np.linalg.norm(direction)
        
        return points
    
    def adaptive_optimization_strategy(initial_points):
        """Apply adaptive optimization with multiple phases"""
        points = initial_points.copy()
        
        # Phase 1: Fast coarse optimization
        # Use a simpler approach with fewer iterations but larger steps
        for i in range(50):
            # Random perturbations
            for j in range(16):
                delta = np.random.normal(0, 0.03, 2)
                test_points = points.copy()
                test_points[j] += delta
                test_points = np.clip(test_points, 0, 1)
                
                current_ratio = compute_min_max_ratio(points)
                test_ratio = compute_min_max_ratio(test_points)
                
                if test_ratio > current_ratio:
                    points = test_points.copy()
        
        # Phase 2: Hierarchical optimization
        points = hierarchical_clustering_optimization(points, max_iter=200)
        
        # Phase 3: Fine-tuned optimization with local search
        for i in range(100):
            # More focused local search around each point
            for j in range(16):
                # Try a more systematic search
                best_points = points.copy()
                best_ratio = compute_min_max_ratio(points)
                
                # Try several small steps in different directions
                for _ in range(20):
                    delta = np.random.normal(0, 0.005, 2)
                    test_points = points.copy()
                    test_points[j] += delta
                    test_points = np.clip(test_points, 0, 1)
                    
                    test_ratio = compute_min_max_ratio(test_points)
                    
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_points = test_points.copy()
                
                points = best_points.copy()
        
        return points
    
    def multi_start_optimization():
        """Run optimization from multiple starting points"""
        configs = create_initial_configurations()
        
        best_points = None
        best_ratio = -float('inf')
        
        # Try each configuration with different optimization approaches
        for name, config in configs.items():
            # Apply adaptive optimization
            optimized = adaptive_optimization_strategy(config)
            
            # Evaluate
            ratio = compute_min_max_ratio(optimized)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized.copy()
        
        # Final fine-tune with L-BFGS-B
        if best_points is not None:
            try:
                x0 = best_points.flatten()
                bounds = [(0, 1) for _ in range(32)]
                
                # Use L-BFGS-B with tight tolerances for final refinement
                result = minimize(
                    lambda x: -compute_min_max_ratio(x.reshape(-1, 2)),  # negative because we maximize
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    final_points = np.clip(final_points, 0, 1)
                    final_ratio = compute_min_max_ratio(final_points)
                    
                    if final_ratio > best_ratio:
                        best_points = final_points
            except Exception:
                pass
        
        return best_points if best_points is not None else configs['regular_polygon']
    
    # Execute the multi-start optimization strategy
    optimized_points = multi_start_optimization()
    
    return optimized_points


# EVOLVE-BLOCK-END
