# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a robust GLS penalty strategy.
    
    This implementation follows proven principles from successful GLS-TSP approaches:
    1. Smooth penalty transitions based on usage counts
    2. Adaptive scaling based on maximum usage in history
    3. Explicit treatment of tour edges
    4. Numerical stability with reasonable bounds
    
    Args:
        edge_distance: np.ndarray of shape (n, n) - original edge distance matrix.
        local_opt_tour: np.ndarray of node IDs in the current local optimal tour.
        edge_n_used: np.ndarray of shape (n, n) - count of how many times each
            edge has been used during perturbation (penalty matrix).

    Returns:
        updated_edge_distance: np.ndarray of shape (n, n) - modified distance matrix
            that guides the local search away from the current local optimum.
    """
    # Create a copy to avoid modifying the original
    updated_edge_distance = edge_distance.copy().astype(float)
    
    # Calculate global statistics for adaptive scaling
    max_usage = np.max(edge_n_used) if np.max(edge_n_used) > 0 else 1.0
    
    # Process each edge in the tour
    n = len(local_opt_tour)
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        
        usage = edge_n_used[a, b]
        
        # Enhanced penalty formula with smooth transitions
        if usage <= 1:
            # Light penalty for very low usage
            penalty = 1.0 + usage * 0.8
        elif usage <= 3:
            # Logarithmic scaling for moderate usage
            penalty = 1.0 + np.log(usage + 1) * 1.2
        else:
            # Exponential scaling for high usage to strongly discourage repetition
            penalty = 1.0 + np.exp(usage / 2.0) * 0.2
            
        # Adaptive scaling based on maximum usage in the history
        adaptive_factor = 1.0 + (max_usage / 10.0) * 0.3
        penalty *= adaptive_factor
        
        # Prevent numerical overflow with reasonable bounds
        penalty = min(50.0, penalty)
        
        # Apply penalty to both directions
        updated_edge_distance[a, b] *= penalty
        updated_edge_distance[b, a] *= penalty
    
    # Handle the closing edge of the tour
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    usage = edge_n_used[a, b]
    
    if usage <= 1:
        penalty = 1.0 + usage * 0.8
    elif usage <= 3:
        penalty = 1.0 + np.log(usage + 1) * 1.2
    else:
        penalty = 1.0 + np.exp(usage / 2.0) * 0.2
        
    adaptive_factor = 1.0 + (max_usage / 10.0) * 0.3
    penalty *= adaptive_factor
    penalty = min(50.0, penalty)
    
    updated_edge_distance[a, b] *= penalty
    updated_edge_distance[b, a] *= penalty
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
