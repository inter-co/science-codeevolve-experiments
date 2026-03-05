# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances to escape local optima in Guided Local Search for TSP.
    
    This implementation combines the robustness of the classic GLS approach with
    enhanced penalty mechanisms that provide better diversification without 
    sacrificing computational efficiency.

    Args:
        edge_distance: np.ndarray of shape (n, n) - original edge distance matrix.
        local_opt_tour: np.ndarray of node IDs in the current local optimal tour.
        edge_n_used: np.ndarray of shape (n, n) - count of how many times each
            edge has been used during perturbation (penalty matrix).

    Returns:
        updated_edge_distance: np.ndarray of shape (n, n) - modified distance matrix
            that guides the local search away from the current local optimum.
    """
    # Create a copy of the original distance matrix
    updated_edge_distance = edge_distance.copy().astype(float)
    
    n = len(local_opt_tour)
    
    # Calculate average edge cost in the current tour for adaptive scaling
    tour_edge_costs = []
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        tour_edge_costs.append(edge_distance[a, b])
    tour_edge_costs.append(edge_distance[local_opt_tour[-1], local_opt_tour[0]])
    
    avg_tour_cost = np.mean(tour_edge_costs)
    
    # Determine adaptive scaling factor based on search dynamics
    max_usage = np.max(edge_n_used)
    # Use a more balanced adaptive factor
    adaptive_factor = 1.0 + 0.6 * np.log(1 + max_usage) / (1 + np.log(100)) if max_usage > 0 else 1.0
    
    # Apply enhanced penalty mechanism to each edge in the tour
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        
        # Classic GLS-style penalty with improved scaling
        # Base penalty component: usage count with logarithmic scaling for better control
        base_penalty = 1.0 + np.log(1 + edge_n_used[a, b])
        
        # Cost-based penalty component: longer edges get higher penalties
        # Normalize by average tour cost to make it relative
        cost_factor = 1.0 + 0.5 * (edge_distance[a, b] / (avg_tour_cost + 1e-8))
        
        # Non-linear transformation with controlled exponential growth
        penalty_multiplier = np.exp(adaptive_factor * base_penalty * cost_factor) - 1
        
        # Apply the penalty to both directions to maintain symmetry
        updated_edge_distance[a, b] *= (1 + penalty_multiplier)
        updated_edge_distance[b, a] = updated_edge_distance[a, b]
    
    # Handle the final edge connecting last to first city
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    
    # Apply same penalty mechanism to final edge
    base_penalty = 1.0 + np.log(1 + edge_n_used[a, b])
    cost_factor = 1.0 + 0.5 * (edge_distance[a, b] / (avg_tour_cost + 1e-8))
    penalty_multiplier = np.exp(adaptive_factor * base_penalty * cost_factor) - 1
    
    # Apply the penalty to both directions to maintain symmetry
    updated_edge_distance[a, b] *= (1 + penalty_multiplier)
    updated_edge_distance[b, a] = updated_edge_distance[a, b]
    
    # Ensure symmetry (recommended for TSP)
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
