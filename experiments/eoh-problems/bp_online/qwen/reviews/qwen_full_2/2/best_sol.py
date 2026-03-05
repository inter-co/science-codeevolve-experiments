# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Enhanced scoring function incorporating best practices from state-of-the-art approaches.
    
    This implementation follows the successful pattern from inspirations:
    - Tight fit preference with proper smoothing
    - Multi-level fragmentation penalty 
    - Utilization efficiency consideration
    - Balanced weight distribution
    
    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Primary scoring: tight fit with smoothing to avoid extreme differences
    tight_fit = 1.0 / (remaining + 0.1)
    
    # Multi-level fragmentation penalty (as seen in inspirations)
    # Very small gaps (< 3) get severe penalty
    # Small gaps (3-8) get moderate penalty  
    # Larger gaps get no penalty
    fragmentation_penalty = np.where(
        remaining < 3, 
        0.001,  # Very strong penalty for tiny gaps
        np.where(remaining < 8, 0.1, 1.0)  # Moderate penalty for small gaps
    )
    
    # Utilization efficiency: prefer bins that are moderately full (around 60%)
    # This balances packing density with future flexibility
    utilization = (100 - remaining) / 100.0
    utilization_score = np.exp(-((utilization - 0.6)**2) / 0.04)
    
    # Bonus for bins that can accommodate additional items of same size
    # This promotes reuse and reduces total bins needed
    if item > 0:
        additional_items = remaining // item
        multipacking_bonus = np.minimum(additional_items, 3) / 3.0
    else:
        multipacking_bonus = 0.0
    
    # Freshness bonus: prefer bins that aren't completely full
    # This maintains flexibility for future items
    freshness_bonus = np.minimum(remaining / 20.0, 1.0)
    
    # Combine all components with carefully tuned weights
    # These weights reflect the importance of each factor based on inspiration analysis
    scores = (0.55 * tight_fit + 
              0.25 * fragmentation_penalty + 
              0.15 * utilization_score +
              0.05 * multipacking_bonus +
              0.05 * freshness_bonus)
    
    # Ensure all scores are positive and reasonably bounded
    scores = np.maximum(scores, 1e-10)
    
    return scores

# EVOLVE-BLOCK-END

