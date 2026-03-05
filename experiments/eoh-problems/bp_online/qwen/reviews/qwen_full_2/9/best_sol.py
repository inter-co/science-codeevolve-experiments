# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score a set of bins for assigning an item in online bin packing.

    This function implements a robust scoring mechanism that:
    1. Rewards tight fits (best-fit principle)
    2. Penalizes small gaps that cause fragmentation
    3. Encourages balanced utilization rates
    4. Promotes reuse of partially filled bins
    
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
    # Simpler linear approach from inspiration 1 for better performance
    utilization = (100 - remaining) / 100.0
    # Prefer moderate utilization (around 50-70%) to balance tight fits and future flexibility
    utilization_score = np.maximum(0, 1 - np.abs(utilization - 0.6) * 2)
    
    # Bonus for bins that can accommodate additional items of same size
    # This encourages reuse of bins and reduces total bin count
    multipacking_bonus = np.zeros_like(bins)
    if item > 0:
        additional_items = remaining // item
        multipacking_bonus = np.minimum(additional_items, 3) / 3.0
    
    # Add a component to prefer bins that are not completely full (to avoid premature filling)
    # This helps maintain flexibility for future items
    freshness_bonus = np.minimum(remaining / 20.0, 1.0)
    
    # Combine components with optimized weights (based on better performing inspiration)
    scores = (0.55 * tight_fit + 
              0.25 * fragmentation_penalty + 
              0.15 * utilization_score +
              0.05 * multipacking_bonus +
              0.05 * freshness_bonus)
    
    # Ensure all scores are non-negative
    scores = np.maximum(scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END

