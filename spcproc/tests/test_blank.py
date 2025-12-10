import pytest
import pandas as pd
import numpy as np

from spcproc.core.blank import BlankProcessor
from spcproc.config import BlankConfig

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_test_data(n_points=1000, value=1.0):
    """Generates a clean grid from 4000 to 400."""
    wn = np.linspace(4000, 400, n_points)
    df = pd.DataFrame({
        "Wavenumber": wn,
        "absorbance": np.full_like(wn, value)
    })
    return df

# ----------------------------------------------------------------------
# 1. Initialization & Region Logic
# ----------------------------------------------------------------------

def test_region_splitting_logic():
    """
    Verify that the processor correctly zeroes out data outside
    the high/mid/low regions defined in config.
    """
    # Create blank with constant value 1.0
    blank_df = make_test_data(value=1.0)
    
    cfg = BlankConfig(
        low_max_wn=800.0,
        high_min_wn=2500.0,
        high_cut_wn=3500.0
    )
    bp = BlankProcessor(blank_df, config=cfg)

    # 1. Test High Cutoff (Global processing)
    # Wavenumber > 3500 should be 0
    high_cut_mask = bp.blank_processed["Wavenumber"] > 3500
    assert np.all(bp.blank_processed.loc[high_cut_mask, "absorbance"] == 0.0)

    # 2. Test Low Region (Should be 0 where Wn > 800)
    low_vals = bp.low_blank
    assert low_vals.loc[low_vals["Wavenumber"] > 800, "absorbance"].sum() == 0.0
    assert low_vals.loc[low_vals["Wavenumber"] <= 800, "absorbance"].sum() > 0

    # 3. Test Mid Region (Should be 0 outside 800-2500)
    mid_vals = bp.mid_blank
    mid_mask = (mid_vals["Wavenumber"] > 800) & (mid_vals["Wavenumber"] < 2500)
    assert mid_vals.loc[~mid_mask, "absorbance"].sum() == 0.0
    assert mid_vals.loc[mid_mask, "absorbance"].sum() > 0


def test_auto_rename_column():
    """It should accept a single column named 'something_else' as absorbance."""
    df = pd.DataFrame({
        "Wavenumber": [1000, 2000],
        "my_weird_col_name": [0.1, 0.2]
    })
    bp = BlankProcessor(df)
    assert "absorbance" in bp.blank_processed.columns


# ----------------------------------------------------------------------
# 2. Math & Subtraction Logic (The Core)
# ----------------------------------------------------------------------

def test_subtraction_logic_deterministic():
    """
    Construct a scenario where we know the exact expected math result.
    
    Setup:
    - Blank = 1.0 everywhere.
    - Sample = 2.0 everywhere.
    
    Expected Behavior:
    1. High Region (> 2500):
       Direct subtraction -> Sample - Blank = 2.0 - 1.0 = 1.0
       
    2. Mid/Low Regions (< 2500):
       Scaled subtraction.
       Scaling Factor = Max(Sample) / Max(Blank) = 2.0 / 1.0 = 2.0
       Result = Sample - (Blank * Factor) = 2.0 - (1.0 * 2.0) = 0.0
    """
    blank_df = make_test_data(value=1.0)
    
    # Sample has value 2.0
    wn = blank_df["Wavenumber"].values
    sample_df = pd.DataFrame({
        "Wavenumber": wn,
        "sample": np.full_like(wn, 2.0)
    })

    # Config ensuring our grid covers the scaling windows
    # Default scaling windows are usually around 600-700 (low) and 800-2000 (mid)
    # Our constant data ensures peaks are found everywhere.
    cfg = BlankConfig(high_min_wn=2500.0, low_max_wn=800.0)
    
    bp = BlankProcessor(blank_df, config=cfg)
    result = bp.subtract(sample_df)
    
    res_vals = result["sample"].values
    
    # Check High Region (> 2500) -> Expect 1.0
    # Note: excluding > 3450 because of high_cut_wn default
    high_mask = (wn > 2500) & (wn < 3450)
    assert np.allclose(res_vals[high_mask], 1.0), "High region should be direct subtraction (2-1=1)"

    # Check Mid Region (e.g., 1500) -> Expect 0.0 (Perfect scaling)
    # 800 < Wn < 2500
    mid_mask = (wn > 800) & (wn < 2500)
    assert np.allclose(res_vals[mid_mask], 0.0), "Mid region should be fully scaled out (2 - 1*2 = 0)"

    # Check Low Region (< 800) -> Expect 0.0 (Perfect scaling)
    low_mask = wn < 800
    assert np.allclose(res_vals[low_mask], 0.0), "Low region should be fully scaled out"


def test_scaling_safeguard_epsilon():
    """If blank is 0 in the scaling window, factor should be 0 (no subtraction), not inf."""
    # Blank is 0.0 everywhere
    blank_df = make_test_data(value=0.0)
    sample_df = make_test_data(value=5.0).rename(columns={"absorbance": "s1"})

    bp = BlankProcessor(blank_df)
    result = bp.subtract(sample_df)

    # Logic: Factor = 5.0 / 0.0 -> Safe guard triggers -> Factor = 0.0
    # Result = 5.0 - (0.0 * 0.0) = 5.0
    assert np.allclose(result["s1"], 5.0)


# ----------------------------------------------------------------------
# 3. Input Validation
# ----------------------------------------------------------------------

def test_missing_wavenumber_raises():
    df = pd.DataFrame({"absorbance": [1, 2]})
    with pytest.raises(ValueError, match="must contain 'Wavenumber'"):
        BlankProcessor(df)

def test_mismatched_grids_raises():
    """If sample and blank have different lengths, raise error."""
    blank_df = make_test_data(n_points=10)
    sample_df = make_test_data(n_points=20).rename(columns={"absorbance": "s1"})
    
    bp = BlankProcessor(blank_df)
    
    with pytest.raises(ValueError, match="grids.*do not match"):
        bp.subtract(sample_df)