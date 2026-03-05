# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using an improved GLS penalty strategy.
    
    This implementation combines:
    1. Classic Voudouris & Tsang frequency-based penalties
    2. Tour-specific cost-aware penalties  
    3. Logarithmic scaling for better penalty distribution
    4. Adaptive penalty intensity based on usage levels
    5. Proper symmetry maintenance for TSP compatibility

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
    
    # Ensure edge_n_used is symmetric (as it should be from GLS history)
    edge_n_used = (edge_n_used + edge_n_used.T) / 2
    
    # Create penalty matrix with same shape as edge_distance
    penalty_matrix = np.zeros((n, n))
    
    # Build list of tour edges (sorted to ensure consistency)
    tour_edges = []
    for i in range(n - 1):
        a, b = local_opt_tour[i], local_opt_tour[i + 1]
        tour_edges.append((min(a, b), max(a, b)))
    a, b = local_opt_tour[-1], local_opt_tour[0]
    tour_edges.append((min(a, b), max(a, b)))
    
    # Create binary matrix for tour edges
    tour_mask = np.zeros((n, n), dtype=bool)
    for a, b in tour_edges:
        tour_mask[a, b] = tour_mask[b, a] = True
    
    # Calculate penalty for each edge based on usage and tour status
    # Classic GLS approach: penalize edges by (cost * usage) / (1 + usage)
    # But enhance with logarithmic scaling and tour prioritization
    
    # Compute penalty components
    max_used = np.max(edge_n_used)
    
    # Base penalty: frequency-based (Voudouris & Tsang style)
    # Use log scaling to prevent extremely large penalties for heavily used edges
    if max_used > 0:
        # Apply log scaling to usage counts to moderate penalty growth
        log_usage = np.log(1 + edge_n_used)
        base_penalty = log_usage / (1.0 + log_usage)
    else:
        base_penalty = np.zeros((n, n))
    
    # Tour-specific enhancement: give higher penalties to tour edges
    tour_penalty = tour_mask.astype(float) * 2.0  # Double penalty for tour edges
    
    # Cost-based component: edges with higher costs should be penalized more
    # This helps escape low-cost local optima
    cost_penalty = np.zeros((n, n))
    for a, b in tour_edges:
        cost_penalty[a, b] = edge_distance[a, b]
        cost_penalty[b, a] = edge_distance[a, b]
    
    # Normalize costs to [0,1] range for fair weighting
    max_cost = np.max(cost_penalty) + 1e-8
    cost_penalty = cost_penalty / max_cost
    
    # Combine all penalty components with appropriate weights
    # This balances frequency, tour status, and cost information
    penalty_matrix = base_penalty + tour_penalty + 0.5 * cost_penalty
    
    # Apply adaptive scaling based on overall usage level
    # More aggressive penalties when edges are heavily used
    if max_used > 0:
        # Scale penalty intensity based on how much we've used edges
        intensity_factor = 1.0 + 0.5 * np.log(1 + max_used)
        penalty_matrix *= intensity_factor
    
    # Apply penalties to the distance matrix
    # Use multiplicative approach to preserve relative distances
    # Add small constant to avoid zero penalties
    epsilon = 1e-8
    penalty_factor = 1.0 + penalty_matrix
    
    # Apply penalty factor to edge distances
    updated_edge_distance = updated_edge_distance * penalty_factor
    
    # Ensure minimum increase to maintain reasonable exploration
    min_increase = 1.1
    updated_edge_distance = np.maximum(updated_edge_distance, edge_distance * min_increase)
    
    # Maintain symmetry for TSP compatibility
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2
    
    # Ensure reasonable numerical bounds
    updated_edge_distance = np.clip(updated_edge_distance, 1e-6, 1e6)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
