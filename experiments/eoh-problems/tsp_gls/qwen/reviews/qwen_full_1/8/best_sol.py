# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """Update edge distances using a refined GLS penalty strategy.
    
    This implementation refines the approach by:
    1. Using the proven structure from inspiration 1
    2. Improving vectorization for better performance
    3. Adding careful bounds checking for numerical stability
    4. Maintaining clean, deterministic behavior

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
    
    # Process each edge in the tour with vectorized operations for efficiency
    n = len(local_opt_tour)
    tour_edges_i = local_opt_tour[:-1]
    tour_edges_j = local_opt_tour[1:]
    
    # Handle the final edge connecting back to start
    tour_edges_i = np.append(tour_edges_i, local_opt_tour[-1])
    tour_edges_j = np.append(tour_edges_j, local_opt_tour[0])
    
    # Get usage counts for all tour edges
    usage_counts = edge_n_used[tour_edges_i, tour_edges_j]
    
    # Apply penalty scaling - refined version of inspiration 1 approach
    penalties = np.ones_like(usage_counts)
    
    # Light penalty for low usage (0-1)
    light_mask = usage_counts <= 1
    penalties[light_mask] = 1.0 + usage_counts[light_mask] * 0.8
    
    # Moderate penalty for medium usage (1-3) using logarithmic scaling
    moderate_mask = (usage_counts > 1) & (usage_counts <= 3)
    penalties[moderate_mask] = 1.0 + np.log(usage_counts[moderate_mask] + 1) * 1.2
    
    # Heavy penalty for high usage (>3) using exponential scaling
    heavy_mask = usage_counts > 3
    penalties[heavy_mask] = 1.0 + np.exp(usage_counts[heavy_mask] / 2.0) * 0.2
    
    # Apply adaptive factor based on maximum usage (similar to inspiration 1)
    adaptive_factor = 1.0 + (max_usage / 10.0) * 0.3
    penalties *= adaptive_factor
    
    # Apply hard caps to prevent extreme inflation
    penalties = np.minimum(penalties, 50.0)
    
    # Apply penalties (both directions)
    updated_edge_distance[tour_edges_i, tour_edges_j] *= penalties
    updated_edge_distance[tour_edges_j, tour_edges_i] = updated_edge_distance[tour_edges_i, tour_edges_j]
    
    # Ensure we don't create negative distances
    updated_edge_distance = np.maximum(updated_edge_distance, edge_distance)
    
    return updated_edge_distance
# EVOLVE-BLOCK-END
