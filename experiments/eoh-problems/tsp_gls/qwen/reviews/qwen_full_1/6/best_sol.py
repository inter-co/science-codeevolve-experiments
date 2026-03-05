# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances to escape local optima in Guided Local Search for TSP.

    Implements a robust penalty strategy inspired by successful GLS approaches:
    1. Piecewise penalty scaling (linear, log, exponential) based on usage counts
    2. Adaptive scaling with maximum usage history
    3. Symmetry preservation and numerical stability
    4. Efficient vectorized computation

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
        
        # Penalty formula inspired by successful GLS implementations
        # Uses smooth transitions between penalty regimes
        if usage <= 1:
            # Light penalty for very low usage
            penalty = 1.0 + usage * 0.5
        elif usage <= 5:
            # Logarithmic scaling for moderate usage
            penalty = 1.0 + np.log(usage + 1) * 0.8
        else:
            # Exponential scaling for high usage to strongly discourage repetition
            penalty = 1.0 + np.exp(usage / 3.0) * 0.1
            
        # Adaptive scaling based on maximum usage in the history
        # Helps balance exploration vs exploitation
        adaptive_factor = 1.0 + (max_usage / 15.0) * 0.2
        penalty *= adaptive_factor
        
        # Prevent numerical overflow with reasonable bounds
        penalty = min(100.0, penalty)
        
        # Apply penalty to both directions
        updated_edge_distance[a, b] *= penalty
        updated_edge_distance[b, a] *= penalty
    
    # Handle the closing edge of the tour
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    usage = edge_n_used[a, b]
    
    if usage <= 1:
        penalty = 1.0 + usage * 0.5
    elif usage <= 5:
        penalty = 1.0 + np.log(usage + 1) * 0.8
    else:
        penalty = 1.0 + np.exp(usage / 3.0) * 0.1
        
    adaptive_factor = 1.0 + (max_usage / 15.0) * 0.2
    penalty *= adaptive_factor
    penalty = min(100.0, penalty)
    
    updated_edge_distance[a, b] *= penalty
    updated_edge_distance[b, a] *= penalty
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
