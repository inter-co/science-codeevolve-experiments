# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using an efficient hybrid penalty strategy for GLS in TSP.

    This implementation combines:
    1. Vectorized operations for efficiency
    2. Logarithmic penalty scaling for balanced guidance
    3. Adaptive scaling based on usage history
    4. Proper symmetry maintenance

    Args:
        edge_distance: np.ndarray of shape (n, n) - original edge distance matrix.
        local_opt_tour: np.ndarray of node IDs in the current local optimal tour.
        edge_n_used: np.ndarray of shape (n, n) - count of how many times each
            edge has been used during perturbation (penalty matrix).

    Returns:
        updated_edge_distance: np.ndarray of shape (n, n) - modified distance matrix
            that guides the local search away from the current local optimum.
    """
    n = len(local_opt_tour)
    
    # Create a copy of the original distance matrix
    updated_edge_distance = edge_distance.copy().astype(float)
    
    # Calculate tour statistics for penalty scaling
    tour_edges = np.empty(n, dtype=int)
    tour_edges[:-1] = local_opt_tour[:-1]
    tour_edges[-1] = local_opt_tour[0]
    
    # Vectorized calculation of tour edge costs
    tour_costs = edge_distance[tour_edges[:-1], tour_edges[1:]]
    tour_cost = np.sum(tour_costs)
    avg_tour_edge_cost = tour_cost / n if n > 0 else 1.0
    
    # Calculate adaptive penalty scaling factor based on maximum usage
    max_usage = np.max(edge_n_used)
    adaptive_factor = 1.0 + 0.5 * np.log(1.0 + max_usage) if max_usage > 0 else 1.0
    
    # Create indices for all tour edges
    tour_indices_i = tour_edges[:-1]
    tour_indices_j = tour_edges[1:]
    
    # Apply hybrid penalty to each tour edge using vectorized operations
    tour_usage_counts = edge_n_used[tour_indices_i, tour_indices_j]
    
    # Base penalty from usage with logarithmic scaling
    usage_penalties = 1.0 + np.log(1.0 + tour_usage_counts)
    
    # Tour-specific penalty based on edge cost relative to average tour edge
    tour_penalties = 1.0 + (tour_costs / np.maximum(avg_tour_edge_cost, 1e-8))
    
    # Combined penalty with non-linear transformation
    combined_penalties = (usage_penalties * tour_penalties) ** 1.3
    
    # Apply adaptive scaling factor
    final_penalties = adaptive_factor * combined_penalties
    
    # Apply penalties to both directions of each tour edge
    updated_edge_distance[tour_indices_i, tour_indices_j] *= final_penalties
    updated_edge_distance[tour_indices_j, tour_indices_i] = updated_edge_distance[tour_indices_i, tour_indices_j]
    
    # Apply memory-aware adjustment to non-tour edges to prevent over-penalization
    # This helps in exploring different parts of the search space
    non_tour_mask = np.ones((n, n), dtype=bool)
    non_tour_mask[np.arange(n), tour_edges] = False
    non_tour_mask[tour_edges, np.arange(n)] = False
    np.fill_diagonal(non_tour_mask, False)
    
    # Apply inverse relationship to reduce penalties for frequently used non-tour edges
    non_tour_usage = edge_n_used[non_tour_mask]
    if len(non_tour_usage) > 0:
        memory_adjustments = 1.0 / (1.0 + 0.1 * non_tour_usage)
        updated_edge_distance[non_tour_mask] *= memory_adjustments
    
    # Ensure symmetry of the distance matrix using efficient method
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    # Add small constant to ensure no zero distances for numerical stability
    updated_edge_distance += 1e-8
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
