# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using an efficient hybrid penalty strategy for GLS in TSP.

    This implementation efficiently combines usage-based penalties with tour-specific
    considerations using vectorized operations for optimal performance while maintaining
    numerical stability.

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
    
    # Vectorized extraction of tour edges
    tour_a = local_opt_tour[:-1]
    tour_b = local_opt_tour[1:]
    # Add the closing edge
    tour_a = np.append(tour_a, local_opt_tour[-1])
    tour_b = np.append(tour_b, local_opt_tour[0])
    
    # Get edge costs and usage counts for all tour edges at once
    tour_costs = edge_distance[tour_a, tour_b]
    usage_counts = edge_n_used[tour_a, tour_b]
    
    # Calculate adaptive scaling factor based on maximum usage
    max_usage = np.max(edge_n_used)
    adaptive_factor = 1.0 + 0.5 * np.log(1.0 + max(0, max_usage))
    
    # Apply penalty using vectorized operations
    # Base penalty from usage with logarithmic scaling
    usage_penalty = 1.0 + np.log(1.0 + usage_counts)
    
    # Tour-specific penalty based on edge cost relative to average
    avg_tour_edge_cost = np.mean(tour_costs) if len(tour_costs) > 0 else 1.0
    tour_penalty = 1.0 + (tour_costs / max(avg_tour_edge_cost, 1e-8))
    
    # Combined penalty with non-linear transformation
    combined_penalty = (usage_penalty * tour_penalty) ** 1.3
    
    # Apply adaptive scaling factor
    final_penalty = adaptive_factor * combined_penalty
    
    # Apply penalty to tour edges
    updated_edge_distance[tour_a, tour_b] *= final_penalty
    updated_edge_distance[tour_b, tour_a] = updated_edge_distance[tour_a, tour_b]
    
    # Ensure symmetry and numerical stability
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    # Add small constant to ensure no zero distances for numerical stability
    updated_edge_distance += 1e-8
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
