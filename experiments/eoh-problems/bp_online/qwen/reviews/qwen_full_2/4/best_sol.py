# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Enhanced bin scoring function combining best-fit intuition with strategic penalties.
    
    This approach improves upon basic best-fit by incorporating:
    - Tight fit preference with smoothing
    - Fragmentation avoidance with tiered penalties  
    - Utilization balance favoring moderate packing
    - Multipacking potential bonus
    - Freshness bonus to maintain packing flexibility

    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Primary scoring: tight fit - prefer bins where item fits tightly
    # Using inverse of remaining space with smoothing to avoid extreme values
    tight_fit = 1.0 / (remaining + 0.1)
    
    # Fragmentation penalty: heavily penalize bins that would leave very small gaps
    # These small gaps make future packing difficult
    fragmentation_penalty = np.where(
        remaining < 3, 
        0.001,  # Very strong penalty for tiny gaps
        np.where(remaining < 8, 0.1, 1.0)  # Moderate penalty for small gaps
    )
    
    # Utilization efficiency: prefer bins that are moderately full
    # This helps maintain packing density without creating overly full bins
    utilization = (100 - remaining) / 100.0
    # Use a Gaussian-like function to prefer utilization around 60%
    utilization_score = np.exp(-((utilization - 0.6)**2) / 0.04)
    
    # Bonus for bins that can accommodate additional items of same size
    # This encourages reuse of bins and reduces total bin count
    if item > 0:
        additional_items = remaining // item
        multipacking_bonus = np.minimum(additional_items, 3) / 3.0
    else:
        multipacking_bonus = 0.0
    
    # Add a component to prefer bins that are not completely full (to avoid premature filling)
    # This helps maintain flexibility for future items
    freshness_bonus = np.minimum(remaining / 20.0, 1.0)
    
    # Add a component to prefer bins that are not overly full (maintain packing flexibility)
    # This prevents bins from being filled to near capacity early on
    not_overfull = np.where(remaining > 10, 1.0, 0.5)
    
    # Combine components with optimized weights
    scores = (0.50 * tight_fit + 
              0.25 * fragmentation_penalty + 
              0.15 * utilization_score +
              0.05 * multipacking_bonus +
              0.05 * freshness_bonus +
              0.05 * not_overfull)
    
    # Ensure all scores are non-negative
    scores = np.maximum(scores, 1e-10)
    
    # Normalize scores to be between 0 and 1 for consistency
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score
    
    return scores

# EVOLVE-BLOCK-END

