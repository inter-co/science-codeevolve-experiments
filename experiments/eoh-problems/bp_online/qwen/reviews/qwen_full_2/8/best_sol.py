# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score a set of bins for assigning an item in online bin packing.

    This implementation uses a multi-factor approach that considers:
    1. Tight fit (how much space is left after placing the item)
    2. Space efficiency (fraction of capacity utilized)
    3. Future packing potential (avoiding bins that leave very little room)
    4. Balance of bin utilization (prefer bins that aren't too full or too empty)

    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    # Handle edge case
    if len(bins) == 0:
        return np.array([])
        
    # Calculate various metrics for each bin
    remaining = bins - item
    utilization = (100 - remaining) / 100.0  # Fraction of capacity used
    
    # Factor 1: Tight fit - prefer bins where item fits most tightly
    # Use reciprocal of remaining space (but avoid division by zero)
    tight_fit_score = 1.0 / (remaining + 0.1)
    
    # Factor 2: Space efficiency - prefer bins that utilize capacity well
    # Avoid bins that are either too empty (<20%) or too full (>90%)
    efficiency_factor = 1.0 - np.abs(utilization - 0.5)  # Peak at 50% utilization
    
    # Factor 3: Future packing potential - avoid bins that leave very little room
    # Penalize bins that would leave less than 10 units of space
    future_potential = np.ones_like(remaining)
    future_potential[remaining < 10] = 0.1  # Significantly penalize very small remaining space
    
    # Factor 4: Balance - prefer bins that aren't extremely full or empty
    balance_score = 1.0 - np.abs(utilization - 0.5)
    
    # Combine all factors with weights
    # Tight fit: 40%, Efficiency: 30%, Future Potential: 20%, Balance: 10%
    scores = (0.4 * tight_fit_score + 
              0.3 * efficiency_factor * future_potential + 
              0.2 * balance_score + 
              0.1 * (1.0 / (remaining + 1.0)))
    
    return scores

# EVOLVE-BLOCK-END

