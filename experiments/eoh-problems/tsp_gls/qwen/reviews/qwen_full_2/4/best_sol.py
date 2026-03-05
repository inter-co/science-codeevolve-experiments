# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a robust hybrid penalty strategy for GLS-TSP.
    
    This implementation combines:
    1. Logarithmic penalty scaling (inspiration 1)
    2. Strategic tour edge prioritization (inspiration 2)
    3. Temporal decay for usage recovery
    4. Efficient vectorized operations
    5. Balanced penalty components for stability

    Args:
        edge_distance: np.ndarray of shape (n, n) - original edge distance matrix.
        local_opt_tour: np.ndarray of node IDs in the current local optimal tour.
        edge_n_used: np.ndarray of shape (n, n) - count of how many times each
            edge has been used during perturbation (penalty matrix).

    Returns:
        updated_edge_distance: np.ndarray of shape (n, n) - modified distance matrix
            that guides the local search away from the current local optimum.
    """
    # Create a copy to avoid modifying the original matrix
    updated_edge_distance = edge_distance.copy()
    
    # Get the number of nodes
    n = len(local_opt_tour)
    
    # Create tour edge mask more efficiently using vectorized operations
    tour_edges = []
    tour_indices = np.arange(n)
    next_indices = np.roll(tour_indices, -1)
    tour_pairs = np.column_stack([local_opt_tour[tour_indices], local_opt_tour[next_indices]])
    
    # Set both directions of tour edges
    tour_mask = np.zeros((n, n), dtype=bool)
    tour_mask[tour_pairs[:, 0], tour_pairs[:, 1]] = True
    tour_mask[tour_pairs[:, 1], tour_pairs[:, 0]] = True
    
    # Normalize usage counts for penalty calculation
    max_usage = np.max(edge_n_used) if np.max(edge_n_used) > 0 else 1
    normalized_usage = edge_n_used / max_usage
    
    # Apply logarithmic penalty scaling (similar to inspiration 1 but with tuning)
    # This emphasizes rare usage while providing reasonable penalties for moderate usage
    usage_penalty = 1.0 + 1.5 * np.log1p(normalized_usage)
    
    # Apply enhanced penalty to tour edges to strongly discourage cycling
    # The tour edges are more likely to trap the search, so they get stronger penalties
    tour_multiplier = 1.0 + 2.0 * tour_mask  # Increased multiplier for tour edges
    
    # Combine penalties with careful balancing
    penalty_factor = usage_penalty * tour_multiplier
    
    # Add temporal decay to prevent over-penalization of edges that were used recently
    # This allows the algorithm to recover and re-explore edges that may be beneficial later
    temporal_decay = 0.95  # Moderate decay rate for better balance
    decay_factor = temporal_decay ** (edge_n_used)
    penalty_factor = penalty_factor * decay_factor
    
    # Apply penalties to both directions to maintain symmetry
    updated_edge_distance = edge_distance * penalty_factor
    
    # Ensure symmetry by averaging with transpose
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2
    
    # Ensure no negative distances (shouldn't happen with our construction, but safety)
    updated_edge_distance = np.maximum(updated_edge_distance, edge_distance)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
