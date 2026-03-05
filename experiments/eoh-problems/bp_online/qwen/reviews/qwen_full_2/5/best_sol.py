# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Enhanced bin scoring function for online bin packing.
    
    This scoring function combines proven heuristics:
    1. Tight fit preference (minimize remaining space)
    2. Gap penalty for very small remaining capacities
    3. Utilization balancing for future packing flexibility
    4. Diversity promotion to avoid packing convergence
    
    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    if len(bins) == 0:
        return np.array([])
    
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Base score: inverse of remaining space (tight fit preference)
    # Add small epsilon to avoid division by zero
    base_score = 1.0 / (remaining + 1e-8)
    
    # Penalty for bins that would leave very small gaps (less than 3 units)
    # These gaps are typically unusable for future items
    gap_penalty = np.where(remaining < 3, 0.1, 1.0)
    
    # Prefer bins with moderate utilization (30-70% filled) for better future packing
    # This provides good balance between tight packing and future flexibility
    utilization = (100.0 - remaining) / 100.0
    # Gaussian penalty centered around 0.5 (moderate utilization)
    utilization_score = np.exp(-((utilization - 0.5)**2) / (2 * 0.15**2))
    
    # Diversity bonus: encourage bins that aren't at extreme utilization levels
    # This prevents getting stuck with only very tight or very loose bins
    if len(bins) > 1:
        # Use a simple distance from median approach for diversity
        median_util = np.median(utilization)
        distance_from_median = np.abs(utilization - median_util)
        # Encourage bins that are reasonably distant from median
        diversity_bonus = 1.0 + np.clip(distance_from_median / 0.3, 0, 0.5)
    else:
        diversity_bonus = 1.0
    
    # Combine all components with carefully tuned weights
    # The weights are chosen to balance tight packing with flexibility
    scores = base_score * gap_penalty * utilization_score * diversity_bonus
    
    # Apply additional penalty for bins that are nearly full (leave < 1 unit)
    # to prevent creating bins that can't accommodate anything else
    full_bin_penalty = np.where(remaining < 1, 0.01, 1.0)
    scores *= full_bin_penalty
    
    # Ensure minimum score to avoid numerical issues
    scores = np.maximum(scores, 1e-10)
    
    return scores

# EVOLVE-BLOCK-END

