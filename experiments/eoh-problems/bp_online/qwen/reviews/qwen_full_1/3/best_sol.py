# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Clean, efficient scoring function for online bin packing.
    
    Based on proven heuristics with mathematical rigor:
    - Prioritizes tight fit for immediate efficiency
    - Penalizes fragmentation from small gaps
    - Balances utilization and future flexibility
    - Adapts weights based on item size
    
    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    # Handle edge case of no bins
    if len(bins) == 0:
        return np.array([])
    
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Core scoring components from inspiration programs
    bin_capacity = 100
    
    # 1. Tight fit scoring (primary criterion)
    # Reward bins that minimize immediate waste - inverse of remaining space
    tight_fit = 1.0 / (remaining + 1.0)
    
    # 2. Utilization efficiency  
    # Prefer bins that are moderately filled to maintain packing flexibility
    utilization = (bin_capacity - remaining) / bin_capacity
    # Quadratic penalty for very low or very high utilization
    utilization_score = 1.0 - (utilization - 0.5)**2
    
    # 3. Fragmentation avoidance
    # Penalize bins that would create very small gaps (wasteful)
    # Use exponential penalty for gaps < 10 units
    small_gap_penalty = np.where(
        remaining < 10,
        np.exp(-remaining / 3.0),
        1.0
    )
    
    # 4. Future flexibility
    # Leave adequate space for similar-sized items
    min_remaining = max(5, item // 2)
    flexibility_score = np.where(
        remaining >= min_remaining,
        1.0,
        remaining / min_remaining
    )
    
    # 5. Item-size adaptive weighting
    # Large items: prioritize tight fit and utilization
    # Small items: prioritize flexibility and avoiding waste
    item_ratio = item / bin_capacity
    adaptive_weight = 0.4 + 0.6 * item_ratio  # 0.4 for small items, 1.0 for large items
    
    # Combine all components with carefully tuned weights
    scores = (
        adaptive_weight * tight_fit +
        (1 - adaptive_weight) * utilization_score +
        0.15 * small_gap_penalty +
        0.15 * flexibility_score
    )
    
    # Ensure numerical stability and positive scores
    scores = np.maximum(scores, 1e-8)
    
    # Normalize scores to prevent extreme variations (important for stable selection)
    if len(scores) > 1:
        # Use robust scaling to maintain relative differences while preventing outliers
        median_score = np.median(scores)
        if median_score > 0:
            scores = scores / median_score
            # Clamp extreme values to prevent numerical issues
            scores = np.clip(scores, 0.01, 10.0)
    
    return scores

# EVOLVE-BLOCK-END

