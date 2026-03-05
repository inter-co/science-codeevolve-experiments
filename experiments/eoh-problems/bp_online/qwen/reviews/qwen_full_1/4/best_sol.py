# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Optimized scoring function for online bin packing.
    
    Combines the best elements from state-of-the-art approaches:
    - Tight fit priority (like Best Fit)
    - Fragmentation avoidance (penalizing small gaps)
    - Adaptive weighting based on item size
    - Clean, efficient vectorized implementation
    
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
    
    # Core scoring components
    bin_capacity = 100
    
    # 1. Tight fit scoring (primary objective) - reward bins with less remaining space
    # Using reciprocal to emphasize bins with less remaining capacity
    tight_fit = 1.0 / (remaining + 1.0)
    
    # 2. Balance scoring - prefer moderate utilization to maintain packing flexibility
    utilization = (bin_capacity - remaining) / bin_capacity
    # Prefer moderate utilization (around 50%) to avoid both empty and full bins
    balance = 1.0 - np.abs(utilization - 0.5)
    
    # 3. Fragmentation penalty - discourage very small gaps that waste space
    # Use a more aggressive penalty for gaps < 5 units
    gap_penalty = np.where(
        remaining < 5,
        np.exp(-remaining / 1.5),  # More aggressive penalty than previous version
        1.0
    )
    
    # 4. Future flexibility - leave enough space for similar items
    # For larger items, require more space; for smaller items, be more flexible
    min_remaining = max(5, item // 2)
    flexibility = np.where(
        remaining >= min_remaining,
        1.0,
        remaining / min_remaining
    )
    
    # 5. Adaptive weighting based on item size
    # Smaller items: prioritize flexibility and avoiding waste
    # Larger items: prioritize tight fit and efficient use of space
    item_ratio = item / bin_capacity
    tight_fit_weight = 0.6 + 0.4 * item_ratio  # 0.6 for small, 1.0 for large items
    
    # Combine components with optimized weights
    scores = (
        tight_fit_weight * tight_fit +
        (1 - tight_fit_weight) * balance +
        0.15 * gap_penalty +
        0.15 * flexibility
    )
    
    # Ensure numerical stability and positive scores
    scores = np.maximum(scores, 1e-10)
    
    return scores

# EVOLVE-BLOCK-END

