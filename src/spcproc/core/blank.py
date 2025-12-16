from typing import Optional, Tuple
import numpy as np
import pandas as pd

from spcproc.config import BlankConfig


def process_blank_core(
    blank_df: pd.DataFrame,
    config: BlankConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split blank spectrum into processed, high, low, and mid region components.
    """
    if "Wavenumber" not in blank_df.columns:
        raise ValueError("Blank DataFrame must contain 'Wavenumber' column.")

    blank_df = blank_df.copy()

    # Normalize intensity column name
    if "absorbance" not in blank_df.columns:
        other_cols = [c for c in blank_df.columns if c != "Wavenumber"]
        if len(other_cols) == 1:
            blank_df.rename(columns={other_cols[0]: "absorbance"}, inplace=True)
        else:
            raise ValueError("Blank DataFrame must contain exactly one absorbance column.")

    # 1. Processed blank (cutoff > 3450)
    blank_proc = blank_df.copy()
    blank_proc.loc[blank_proc["Wavenumber"] > config.high_cut_wn, "absorbance"] = 0.0

    low_max = float(config.low_max_wn)
    high_min = float(config.high_min_wn)

    # 2. High region (> 2500)
    high = blank_proc.copy()
    high["absorbance"] = np.where(
        high["Wavenumber"] <= high_min, 0.0, high["absorbance"]
    )

    # 3. Low region (<= 810)
    low = blank_proc.copy()
    low["absorbance"] = np.where(
        low["Wavenumber"] > low_max, 0.0, low["absorbance"]
    )

    # 4. Mid region (810 < wn < 2500)
    mid = blank_proc.copy()
    mid["absorbance"] = np.where(
        (mid["Wavenumber"] > low_max) & (mid["Wavenumber"] < high_min),
        mid["absorbance"],
        0.0,
    )

    return blank_proc, high, low, mid


def blank_subtraction_core(
    baselined_spectra: pd.DataFrame,
    high_blank: pd.DataFrame,
    low_blank: pd.DataFrame,
    mid_blank: pd.DataFrame,
    config: BlankConfig,
) -> pd.DataFrame:
    """
    Subtract blank from spectra.
    High region is direct subtraction; Low/Mid regions use peak-ratio scaling.
    """
    # Validation
    if "Wavenumber" not in baselined_spectra.columns:
        raise ValueError("Input spectra must contain 'Wavenumber' column.")
    
    for name, df in [("high", high_blank), ("low", low_blank), ("mid", mid_blank)]:
        if "Wavenumber" not in df.columns or "absorbance" not in df.columns:
            raise ValueError(f"{name}_blank must contain 'Wavenumber' and 'absorbance'.")

    spectra_corrected = baselined_spectra.copy()

    # Pre-extract numpy arrays for speed
    high_abs = high_blank["absorbance"].values
    low_abs = low_blank["absorbance"].values
    mid_abs = mid_blank["absorbance"].values
    Wn = spectra_corrected["Wavenumber"].values

    if not (len(Wn) == len(high_abs) == len(low_abs) == len(mid_abs)):
        raise ValueError("Wavenumber grids of spectra and blanks do not match.")

    # Masks for scaling regions
    low_region_mask = (Wn > config.low_scale_min_wn) & (Wn < config.low_scale_max_wn)
    mid_region_mask = (Wn > config.mid_scale_min_wn) & (Wn < config.mid_scale_max_wn)

    # Process each sample column
    for col in spectra_corrected.columns:
        if col == "Wavenumber":
            continue
            
        sample_abs = spectra_corrected[col].values

        # 1. Direct subtraction (High)
        subtracted_abs = sample_abs - high_abs

        # 2. Scaled subtraction (Low)
        sample_max_low = np.max(sample_abs[low_region_mask])
        blank_max_low = np.max(low_abs[low_region_mask])
        
        low_factor = 0.0
        if blank_max_low > config.scale_epsilon:
            low_factor = sample_max_low / blank_max_low

        subtracted_abs -= (low_abs * low_factor)

        # 3. Scaled subtraction (Mid)
        sample_max_mid = np.max(sample_abs[mid_region_mask])
        blank_max_mid = np.max(mid_abs[mid_region_mask])

        mid_factor = 0.0
        if blank_max_mid > config.scale_epsilon:
            mid_factor = sample_max_mid / blank_max_mid

        subtracted_abs -= (mid_abs * mid_factor)

        spectra_corrected[col] = subtracted_abs

    return spectra_corrected


class BlankProcessor:
    """Stateful wrapper for blank processing."""

    def __init__(self, blank_df: pd.DataFrame, config: Optional[BlankConfig] = None):
        self.config = config or BlankConfig()
        self.blank_raw = blank_df
        
        (
            self.blank_processed,
            self.high_blank,
            self.low_blank,
            self.mid_blank,
        ) = process_blank_core(blank_df, config=self.config)

    def subtract(self, baselined_spectra: pd.DataFrame) -> pd.DataFrame:
        return blank_subtraction_core(
            baselined_spectra=baselined_spectra,
            high_blank=self.high_blank,
            low_blank=self.low_blank,
            mid_blank=self.mid_blank,
            config=self.config,
        )