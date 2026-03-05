# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a refined GLS penalty strategy for TSP.

    This implementation refines the successful approach from INSPIRATION 2 by 
    using a cleaner penalty formulation that closely matches proven effective patterns
    while maintaining computational efficiency and numerical stability.

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
    
    # Get edge costs for all tour edges at once
    tour_costs = edge_distance[tour_a, tour_b]
    
    # Calculate adaptive scaling factor based on maximum usage
    max_usage = np.max(edge_n_used)
    # Use a simpler, more robust scaling factor that works consistently
    adaptive_factor = 1.0 + 0.6 * np.log(1.0 + max_usage) / (1.0 + np.log(100.0)) if max_usage > 0 else 1.0
    
    # Get usage counts for tour edges
    usage_counts = edge_n_used[tour_a, tour_b]
    
    # Apply classic GLS-style penalty with clean formulation
    # Base penalty: logarithmic scaling of usage counts
    base_penalty = 1.0 + np.log(1.0 + usage_counts)
    
    # Cost-based component: normalize by average tour cost for context
    avg_tour_cost = np.mean(tour_costs) if len(tour_costs) > 0 else 1.0
    cost_factor = 1.0 + 0.5 * (tour_costs / max(avg_tour_cost, 1e-8))
    
    # Apply penalty with controlled exponential growth (cleaner than power scaling)
    penalty_multiplier = np.exp(adaptive_factor * base_penalty * cost_factor) - 1.0
    
    # Apply penalty to tour edges (both directions for symmetry)
    updated_edge_distance[tour_a, tour_b] *= (1.0 + penalty_multiplier)
    updated_edge_distance[tour_b, tour_a] = updated_edge_distance[tour_a, tour_b]
    
    # Ensure symmetry explicitly (recommended for TSP)
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
