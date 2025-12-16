"""
Source of truth for the default reference parameters.

The constants used in `config.py` (like `DEFAULT_ZERO_WN`) aren't random;
they were calculated from Amir's original PSI dataset to match the legacy
index-based R implementation.

Since the instrument setup is fixed, these values are effectively hardcoded constants.

You generally don't need to import this. It's here so we don't lose track of
how those numbers were derived, and to help if we ever need to recalibrate
for a new instrument setup.
"""

from dataclasses import dataclass
from typing import Tuple
import pandas as pd


@dataclass(frozen=True)
class ReferenceParameters:
    """
    Holds the mapping between legacy Python indices and physical Wavenumbers.
    """
    # Indices (0-based) from original script
    zero_index_py: int
    high_mid_index: int
    mid_low_index: int

    # Back-calculated Wavenumbers (cm^-1)
    zero_wn: float
    high_mid_wn: float
    mid_low_wn: float

    @property
    def stitch_bounds(self) -> Tuple[float, float]:
        return (self.high_mid_wn, self.mid_low_wn)


# Calculated from Amir's dataset.
# zero_index_py == mid_low_index == 1379 in this specific setup.
DEFAULT_REFERENCE = ReferenceParameters(
    zero_index_py=1379,
    high_mid_index=999,
    mid_low_index=1379,
    zero_wn=1336.5996,
    high_mid_wn=2069.5114,
    mid_low_wn=1336.5996,
)


class ReferenceExtractor:
    """
    Helper to recompute parameters from a reference dataset.
    Only needed if the instrument resolution or reference spectra change.
    """

    def __init__(self, ref_df: pd.DataFrame):
        if "Wavenumber" not in ref_df.columns:
            raise ValueError("Reference DataFrame must contain 'Wavenumber'.")
        self._wn = ref_df["Wavenumber"].to_numpy()

    def compute_from_indices(
        self,
        zero_index_py: int,
        high_mid_index: int,
        mid_low_index: int,
    ) -> ReferenceParameters:
        """Get wavenumbers corresponding to specific array indices."""
        
        # Simple bounds check
        if not (0 <= max(zero_index_py, high_mid_index, mid_low_index) < len(self._wn)):
             raise IndexError(f"One or more indices out of bounds (size={len(self._wn)}).")

        return ReferenceParameters(
            zero_index_py=zero_index_py,
            high_mid_index=high_mid_index,
            mid_low_index=mid_low_index,
            zero_wn=float(self._wn[zero_index_py]),
            high_mid_wn=float(self._wn[high_mid_index]),
            mid_low_wn=float(self._wn[mid_low_index]),
        )

    def compute_with_default_indices(self) -> ReferenceParameters:
        """Re-run extraction using the known legacy indices."""
        return self.compute_from_indices(
            zero_index_py=DEFAULT_REFERENCE.zero_index_py,
            high_mid_index=DEFAULT_REFERENCE.high_mid_index,
            mid_low_index=DEFAULT_REFERENCE.mid_low_index,
        )