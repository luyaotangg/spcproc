from dataclasses import dataclass
from typing import Optional, Tuple


# Derived from original PSI dataset calculations
DEFAULT_ZERO_WN: float = 1336.5996
DEFAULT_STITCH_BOUNDS: Tuple[float, float] = (2069.5114, 1336.5996)


@dataclass
class BaselineConfig:
    """Configuration for baseline correction."""
    
    zero_wn: Optional[float] = None
    stitch_bounds: Optional[Tuple[float, float]] = None
    
    # Degrees of freedom for smooth.spline (low, mid, high)
    dfs: Tuple[int, int, int] = (10, 10, 7)
    
    # Legacy compatibility
    zero_idx_py: int = 1379


@dataclass
class BlankConfig:
    """
    Configuration for blank subtraction.
    Defaults are tuned for PSI Teflon-filter FTIR spectra (cm^-1).
    """

    # Region definitions
    high_cut_wn: float = 3450.0
    high_min_wn: float = 2500.0
    low_max_wn:  float = 810.0

    # Scaling windows (Low region)
    low_scale_min_wn: float = 580.0
    low_scale_max_wn: float = 700.0

    # Scaling windows (Mid region)
    mid_scale_min_wn: float = 810.0
    mid_scale_max_wn: float = 2000.0

    # Numerical stability
    scale_epsilon: float = 1e-9