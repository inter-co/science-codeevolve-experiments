# EVOLVE-BLOCK-START
import numpy as np

def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Enhanced scoring function incorporating strategic bin utilization and gap management.
    
    This improved version builds upon the polyhedral theory foundation while integrating:
    1. Enhanced gap penalty mechanisms inspired by information theory
    2. Strategic utilization positioning for better long-term packing flexibility
    3. Adaptive weighting based on item size characteristics
    4. Improved numerical stability and tie-breaking mechanisms

    Args:
        item: Size of the current item to be packed.
        bins: Remaining capacities of feasible bins (bins where item fits).

    Returns:
        scores: Array of scores for each feasible bin. Higher score = preferred bin.
    """
    if len(bins) == 0:
        return np.array([0.0])
    
    # Calculate remaining capacity after placing item
    remaining = bins - item
    
    # Prevent negative remaining capacities (shouldn't happen with valid inputs)
    remaining = np.maximum(remaining, 0)
    
    # Enhanced gap penalty - more aggressive for very small gaps
    # This encourages avoiding bins that leave only tiny amounts of space
    gap_penalty = np.exp(-remaining / 3.0)  # Slightly stronger penalty than before
    
    # Strategic utilization positioning - prefer bins with moderate utilization
    # This helps avoid creating either very empty or very full bins
    utilization = (100 - remaining) / 100.0
    # Prefer utilization around 50% - penalize extremes
    utilization_score = 1.0 - (utilization - 0.5)**2
    
    # Packing efficiency - measure how effectively we use bin capacity
    # For items close to bin capacity, we want to fill them up
    # For small items, we want to minimize waste
    item_ratio = item / 100.0
    packing_efficiency = item_ratio * (1.0 - remaining / 100.0)
    
    # Item-bin compatibility - encourage placing items in bins that can accommodate them well
    compatibility_score = np.minimum(remaining / item, 1.0) if item > 0 else 1.0
    
    # Adaptive weighting based on item size characteristics
    # Smaller items benefit more from careful gap management
    # Larger items benefit more from utilization balance
    if item_ratio < 0.1:  # Very small items
        weights = [0.2, 0.3, 0.4, 0.1]  # Emphasize gap management
    elif item_ratio < 0.5:  # Medium items
        weights = [0.3, 0.2, 0.3, 0.2]  # Balanced approach
    else:  # Large items
        weights = [0.2, 0.4, 0.2, 0.2]  # Emphasize utilization
    
    # Calculate combined score with adaptive weights
    raw_scores = (packing_efficiency * weights[0] + 
                  utilization_score * weights[1] + 
                  gap_penalty * weights[2] + 
                  compatibility_score * weights[3])
    
    # Apply a sigmoid-like transformation to smooth out extreme variations
    # This makes the scoring more robust to outliers while maintaining discrimination
    scores = 1.0 / (1.0 + np.exp(-raw_scores * 5.0))  # Sigmoid transformation
    
    # Normalize scores to prevent extreme variations
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / (max_score + 1e-10)
    
    # Ensure numerical stability - avoid exactly zero scores
    scores = np.maximum(scores, 1e-10)
    
    # Add deterministic tie-breaking for identical scores
    # This ensures consistent behavior without introducing randomness
    if len(scores) > 1:
        # Create a small deterministic offset based on bin index
        tie_breaker = np.arange(len(scores)) * 1e-12
        scores = scores + tie_breaker
    
    return scores

# EVOLVE-BLOCK-END

