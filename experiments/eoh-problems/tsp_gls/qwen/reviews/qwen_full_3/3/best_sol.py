# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a refined penalty strategy for GLS-TSP.
    
    This implementation follows a clean approach:
    1. Logarithmic penalty for tour edges based on usage counts
    2. Adaptive scaling factor based on maximum usage
    3. Ensures numerical stability
    4. Maintains symmetry in the distance matrix

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
    updated_edge_distance = edge_distance.copy()
    
    # Get tour length
    n = len(local_opt_tour)
    
    # Calculate adaptive penalty factor based on search progress
    max_usage = np.max(edge_n_used)
    if max_usage > 0:
        # Adaptive scaling: stronger penalties for higher usage, but capped
        penalty_scale = 1.0 + 0.5 * np.log(1 + max_usage)
        penalty_scale = min(penalty_scale, 5.0)  # Cap the penalty scale
    else:
        penalty_scale = 1.0
    
    # Vectorized computation of tour edges
    tour_edges = np.column_stack([local_opt_tour[:-1], local_opt_tour[1:]])
    tour_edges = np.vstack([tour_edges, [local_opt_tour[-1], local_opt_tour[0]]])
    
    # Apply penalty to tour edges with logarithmic scaling
    # This provides non-linear but controlled penalty growth
    tour_usages = edge_n_used[tour_edges[:, 0], tour_edges[:, 1]]
    
    # Base penalty component: logarithmic scaling with usage count
    base_penalty = 1.0 + 0.5 * np.log(1.0 + tour_usages)
    
    # Combine components with adaptive scaling
    final_penalty = penalty_scale * base_penalty
    
    # Apply penalties to both directions of tour edges
    updated_edge_distance[tour_edges[:, 0], tour_edges[:, 1]] *= final_penalty
    updated_edge_distance[tour_edges[:, 1], tour_edges[:, 0]] *= final_penalty
    
    # Ensure symmetry by taking maximum with transpose
    # This maintains the property that if an edge was penalized more in one direction,
    # it's penalized equally in both directions
    updated_edge_distance = np.maximum(updated_edge_distance, updated_edge_distance.T)
    
    # Clip extreme values to maintain numerical stability
    max_allowed = np.max(edge_distance) * 1000.0
    updated_edge_distance = np.clip(updated_edge_distance, 0, max_allowed)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
