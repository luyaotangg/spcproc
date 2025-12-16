from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

from spcproc.config import BaselineConfig, DEFAULT_ZERO_WN, DEFAULT_STITCH_BOUNDS

try:
    numpy2ri.activate()
except DeprecationWarning:
    pass

_stats = importr("stats")


def zero_shift(df: pd.DataFrame, zero_wn: float) -> pd.DataFrame:
    """
    Zero-shift spectra by subtracting the absorbance at `zero_wn`.
    """
    if "Wavenumber" not in df.columns:
        raise ValueError("DataFrame must contain 'Wavenumber' column.")

    out = df.copy()
    wn = out["Wavenumber"].to_numpy().astype(float)
    
    # Find nearest index
    idx = int(np.argmin(np.abs(wn - zero_wn)))
    
    sample_cols = [c for c in out.columns if c != "Wavenumber"]
    for col in sample_cols:
        out[col] = out[col] - out[col].iloc[idx]

    return out


def assign_weights(Wn: np.ndarray) -> np.ndarray:
    """Assign asymmetric weights following the original R implementation."""
    Wn = np.asarray(Wn, dtype=float)
    w = np.zeros_like(Wn, dtype=float)

    # Specific bands from R script
    w[(Wn < 4500) & (Wn > 3550)] = 10
    w[(Wn < 2160) & (Wn > 1820)] = 4
    w[(Wn < 1500) & (Wn > 1480)] = 30
    w[(Wn < 690)  & (Wn > 680)]  = 10
    w[(Wn < 820)  & (Wn > 800)]  = 10
    w[(Wn < 1330) & (Wn > 1320)] = 10
    w[(Wn < 450)  & (Wn > 300)]  = 10
    w[(Wn < 590)  & (Wn > 580)]  = 10

    return w


def mask_section(Wn: np.ndarray, K: int) -> np.ndarray:
    """
    Section mask for baseline fitting based on Amir's R logic.
    K=1: 400-1380 | K=2: 0-2300 | K=3: 1300-4000
    """
    if K == 1:
        return Wn <= 1380.0
    elif K == 2:
        return (Wn >= 0.0) & (Wn <= 2300.0)
    elif K == 3:
        return Wn >= 1300.0
    else:
        raise ValueError("K must be 1, 2, or 3.")


def smooth_spline_r(x: np.ndarray, y: np.ndarray, df: int, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Wrapper for R's stats::smooth.spline."""
    x_r = robjects.FloatVector(np.asarray(x, dtype=float))
    y_r = robjects.FloatVector(np.asarray(y, dtype=float))

    if w is not None:
        w_r = robjects.FloatVector(np.asarray(w, dtype=float))
        fit = _stats.smooth_spline(x_r, y_r, w=w_r, df=df)
    else:
        fit = _stats.smooth_spline(x_r, y_r, df=df)

    pred = _stats.predict_smooth_spline(fit, x_r)
    # pred[1] is the y component of the prediction
    return np.array(pred[1])


def smooth_baseline_for_column(
    S: np.ndarray, Wn: np.ndarray, w: np.ndarray, df_value: int, K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit baseline for a single column using segment K."""
    mask = mask_section(Wn, K)
    w_eff = w.copy().astype(float)
    w_eff[~mask] = 0.0

    # smooth.spline requires sorted X
    order = np.argsort(Wn)
    
    baseline_sorted = smooth_spline_r(
        Wn[order], 
        S[order], 
        w=w_eff[order], 
        df=df_value
    )

    # Restore original order
    baseline = np.empty_like(baseline_sorted)
    baseline[order] = baseline_sorted
    
    return baseline, S - baseline


def baseline_corrector_core(
    df: pd.DataFrame,
    dfs: Tuple[int, int, int],
    zero_wn: float,
    stitch_bounds: Tuple[float, float],
) -> Dict[str, pd.DataFrame]:
    
    df = df.copy()
    # Ensure standard naming
    df.rename(columns={df.columns[0]: "Wavenumber"}, inplace=True)

    sample_cols = [c for c in df.columns if c != "Wavenumber"]
    Wn = df["Wavenumber"].values.astype(float)

    mid_high_wn, low_mid_wn = stitch_bounds
    
    # Masks for stitching
    low_mask = Wn <= low_mid_wn
    mid_mask = (Wn > low_mid_wn) & (Wn <= mid_high_wn)
    high_mask = Wn > mid_high_wn

    MyData = zero_shift(df, zero_wn=zero_wn)

    # Prepare containers
    baselined_stitched = pd.DataFrame({"Wavenumber": Wn})
    background_stitched = pd.DataFrame({"Wavenumber": Wn})
    
    # Iterate through K segments
    for K in (1, 2, 3):
        baselined_tmp = pd.DataFrame({"Wavenumber": Wn})
        background_tmp = pd.DataFrame({"Wavenumber": Wn})
        w = assign_weights(Wn)

        for col in sample_cols:
            baseline, corrected = smooth_baseline_for_column(
                MyData[col].values, Wn, w, df_value=dfs[K - 1], K=K
            )
            background_tmp[col] = baseline
            baselined_tmp[col] = corrected

        # Stitch logic
        if K == 1:
            baselined_stitched = baselined_tmp.copy()
            background_stitched = background_tmp.copy()
        elif K == 2:
            baselined_stitched.loc[mid_mask, sample_cols] = baselined_tmp.loc[mid_mask, sample_cols]
            background_stitched.loc[mid_mask, sample_cols] = background_tmp.loc[mid_mask, sample_cols]
        elif K == 3:
            baselined_stitched.loc[high_mask, sample_cols] = baselined_tmp.loc[high_mask, sample_cols]
            background_stitched.loc[high_mask, sample_cols] = background_tmp.loc[high_mask, sample_cols]

    return {
        "original_spectra": MyData,
        "background_spectra": background_stitched,
        "baselined_spectra": baselined_stitched,
    }


class BaselineCorrector:
    """
    Handles baseline correction configuration and validation.
    Defaults to PSI Teflon-filter settings if config not provided.
    """

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()

    @staticmethod
    def _get_wn_range(df: pd.DataFrame) -> Tuple[float, float]:
        if "Wavenumber" not in df.columns:
            raise ValueError("DataFrame must contain a 'Wavenumber' column.")
        wn = df["Wavenumber"].to_numpy()
        return float(np.min(wn)), float(np.max(wn))

    def _prepare_zero_wn(self, df: pd.DataFrame) -> float:
        wn_min, wn_max = self._get_wn_range(df)
        
        if self.config.zero_wn is not None:
            zero_wn = float(self.config.zero_wn)
            source = "BaselineConfig.zero_wn"
        else:
            zero_wn = float(DEFAULT_ZERO_WN)
            source = f"DEFAULT_ZERO_WN ({DEFAULT_ZERO_WN})"

        if not (wn_min <= zero_wn <= wn_max):
            raise ValueError(
                "Invalid zero-shift reference for this dataset.\n"
                f"  - Requested zero_wn = {zero_wn} (from {source})\n"
                f"  - Data wavenumber range = [{wn_min}, {wn_max}]\n\n"
                "This baseline-correction implementation is tuned for PSI Teflon-filter "
                f"FTIR spectra that cover the reference region around {DEFAULT_ZERO_WN} cm⁻¹.\n"
                "If your spectra do not include this region, this algorithm is likely "
                "not appropriate for your data.\n\n"
                "If you are still using compatible PSI-type spectra but with slightly "
                "different resolution, please set BaselineConfig(zero_wn=...) to a "
                "nearby wavenumber that is actually present in your data."
            )
        return zero_wn

    def _prepare_stitch_bounds(self, df: pd.DataFrame) -> Tuple[float, float]:
        if self.config.stitch_bounds is not None:
            b1, b2 = self.config.stitch_bounds
            source = "BaselineConfig.stitch_bounds"
        else:
            b1, b2 = DEFAULT_STITCH_BOUNDS
            source = f"DEFAULT_STITCH_BOUNDS {DEFAULT_STITCH_BOUNDS}"

        b1, b2 = float(b1), float(b2)
        wn_min, wn_max = self._get_wn_range(df)

        if not (wn_min <= b1 <= wn_max) or not (wn_min <= b2 <= wn_max):
            raise ValueError(
                "Invalid stitch_bounds for this dataset.\n"
                f"  - Requested stitch_bounds = ({b1}, {b2}) (from {source})\n"
                f"  - Data wavenumber range  = [{wn_min}, {wn_max}]\n\n"
                "These stitching points were calibrated for PSI Teflon-filter FTIR "
                "spectra. If your spectra do not cover these regions, this specific "
                "baseline/blank algorithm is not appropriate for your data.\n\n"
                "If you are still within the same measurement setup but with slightly "
                "different resolution, you may tweak BaselineConfig(stitch_bounds=...) "
                "to nearby values that exist in your wavenumber grid."
            )

        if not (b1 > b2):
            raise ValueError(
                f"stitch_bounds={(b1, b2)} are not in the expected order (high > low). "
                "Please check your configuration."
            )

        return (b1, b2)

    def run(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return baseline_corrector_core(
            df=df,
            dfs=self.config.dfs,
            zero_wn=self._prepare_zero_wn(df),
            stitch_bounds=self._prepare_stitch_bounds(df),
        )