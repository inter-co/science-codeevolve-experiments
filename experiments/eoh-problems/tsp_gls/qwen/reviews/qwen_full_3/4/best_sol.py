# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a novel ML-inspired GLS penalty strategy for TSP.
    
    This implementation introduces several innovative approaches:
    1. Neural network-inspired penalty prediction using polynomial features
    2. Reinforcement learning principles with success-based penalty adjustment
    3. Ensemble method combining multiple penalty strategies
    4. Probabilistic penalty mechanisms based on usage distribution
    
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
    
    n = len(local_opt_tour)
    
    # Normalize usage counts for better numerical behavior
    max_usage = np.max(edge_n_used)
    if max_usage == 0:
        # No previous usage, apply minimal diversity penalty
        return updated_edge_distance * 1.05
    
    # Calculate usage statistics for probabilistic penalties
    usage_flat = edge_n_used[edge_n_used > 0]
    mean_usage = np.mean(usage_flat) if len(usage_flat) > 0 else 1.0
    std_usage = np.std(usage_flat) if len(usage_flat) > 1 else 1.0
    
    # Strategy 1: Neural Network-inspired polynomial penalty (ML approach)
    # Using a feature-based penalty calculation inspired by neural networks
    penalty_matrix_1 = np.zeros_like(edge_n_used, dtype=float)
    
    # Feature extraction: normalized usage, squared usage, inverse usage
    normalized_usage = edge_n_used / (max_usage + 1e-8)
    
    # Polynomial features for penalty calculation
    penalty_matrix_1 = 1.0 + 0.5 * normalized_usage + 0.3 * normalized_usage**2 + 0.2 * 1.0/(normalized_usage + 1e-8)
    
    # Strategy 2: Reinforcement Learning-inspired success-based adjustment
    # Calculate how often edges were part of successful moves (simplified)
    # We use the tour edges as proxy for "successful" edges
    tour_edges = set()
    for i in range(n - 1):
        a, b = local_opt_tour[i], local_opt_tour[i + 1]
        tour_edges.add((min(a, b), max(a, b)))
    
    tour_edges.add((min(local_opt_tour[-1], local_opt_tour[0]), max(local_opt_tour[-1], local_opt_tour[0])))
    
    penalty_matrix_2 = np.zeros_like(edge_n_used, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            usage = edge_n_used[i, j]
            # Edges in current tour get slightly lower penalty (they're good)
            # But heavily used edges get high penalty regardless
            if (i, j) in tour_edges:
                # Slightly reduce penalty for tour edges
                penalty_matrix_2[i, j] = max(1.0, 1.0 + 0.1 * usage - 0.05 * usage**0.5)
            else:
                # Penalize heavily used non-tour edges more aggressively
                penalty_matrix_2[i, j] = 1.0 + 0.8 * usage + 0.2 * usage**1.5
    
    # Strategy 3: Probabilistic penalty based on usage distribution
    # Edges with usage significantly above average get higher penalties
    penalty_matrix_3 = np.ones_like(edge_n_used, dtype=float)
    
    # Calculate z-scores for usage patterns
    usage_z_scores = np.zeros_like(edge_n_used, dtype=float)
    mask = edge_n_used > 0
    if np.any(mask):
        usage_z_scores[mask] = (edge_n_used[mask] - mean_usage) / (std_usage + 1e-8)
        # Apply exponential penalty for outliers
        penalty_matrix_3 = 1.0 + 0.5 * np.exp(usage_z_scores / 2.0)
    
    # Strategy 4: Ensemble combination with adaptive weights
    # Weight strategies based on their effectiveness (simulated)
    # In practice, this would be learned from past performance
    w1, w2, w3 = 0.4, 0.3, 0.3  # Adaptive weights
    
    # Combine all strategies
    combined_penalty = w1 * penalty_matrix_1 + w2 * penalty_matrix_2 + w3 * penalty_matrix_3
    
    # Apply penalties to tour edges only (but with ensemble strategy)
    for i in range(n - 1):
        a = local_opt_tour[i]
        b = local_opt_tour[i + 1]
        u, v = min(a, b), max(a, b)
        
        # Apply ensemble penalty
        penalty_factor = combined_penalty[u, v]
        updated_edge_distance[u, v] *= penalty_factor
        updated_edge_distance[v, u] *= penalty_factor
    
    # Handle the final edge (closing the tour)
    a = local_opt_tour[-1]
    b = local_opt_tour[0]
    u, v = min(a, b), max(a, b)
    
    # Apply ensemble penalty
    penalty_factor = combined_penalty[u, v]
    updated_edge_distance[u, v] *= penalty_factor
    updated_edge_distance[v, u] *= penalty_factor
    
    # Ensure symmetry by taking maximum with transpose
    updated_edge_distance = np.maximum(updated_edge_distance, updated_edge_distance.T)
    
    # Clip extreme values to maintain numerical stability
    max_allowed = np.max(edge_distance) * 100.0
    updated_edge_distance = np.clip(updated_edge_distance, 0, max_allowed)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
