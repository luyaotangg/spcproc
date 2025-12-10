import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from spcproc.core.baseline import (
    zero_shift,
    assign_weights,
    mask_section,
    BaselineCorrector,
    baseline_corrector_core,
)
from spcproc.config import BaselineConfig

# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_spectra():
    """Generates a standard dummy spectrum DataFrame."""
    wn = np.linspace(4000, 400, 100)  # Descending order typical for FTIR
    data = {
        "Wavenumber": wn,
        "Sample1": np.sin(wn / 500) + 2.0,  # Offset to keep positive
        "Sample2": np.cos(wn / 500) + 1.5,
    }
    return pd.DataFrame(data)

@pytest.fixture
def config():
    """Default config for testing."""
    return BaselineConfig(
        zero_wn=1380.0,
        stitch_bounds=(2070.0, 1340.0),
        dfs=(5, 5, 5)
    )

# =====================================================================
# Unit Tests: Helper Functions
# =====================================================================

def test_zero_shift_logic():
    """Should subtract the value at the nearest wavenumber from the whole column."""
    df = pd.DataFrame({
        "Wavenumber": [1000.0, 1500.0, 2000.0],
        "S1": [10.0, 20.0, 30.0],
        "S2": [5.0,  5.0,  5.0]
    })
    
    # Target 1500 -> Index 1 -> Offsets are 20.0 and 5.0
    shifted = zero_shift(df, zero_wn=1501.0) # slightly off to test "nearest"
    
    # Check S1
    assert shifted.loc[1, "S1"] == 0.0      # 20 - 20
    assert shifted.loc[0, "S1"] == -10.0    # 10 - 20
    
    # Check S2
    assert shifted["S2"].sum() == 0.0       # All were 5.0, all become 0.0

def test_zero_shift_missing_column():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pytest.raises(ValueError, match="must contain 'Wavenumber'"):
        zero_shift(df, 1000)

@pytest.mark.parametrize("wn, expected_weight", [
    (4000, 10), # 3550 < wn < 4500
    (2000, 4),  # 1820 < wn < 2160
    (1490, 30), # 1480 < wn < 1500
    (2500, 0),  # Gap region
])
def test_assign_weights(wn, expected_weight):
    """Verify weight assignment logic for single points."""
    w = assign_weights(np.array([wn]))
    assert w[0] == expected_weight

@pytest.mark.parametrize("K, wn_val, expected", [
    (1, 1000, True),  # < 1380
    (1, 1400, False), # > 1380
    (2, 2000, True),  # 0-2300
    (2, 3000, False), # > 2300
    (3, 3000, True),  # > 1300
    (3, 1000, False), # < 1300
])
def test_mask_section(K, wn_val, expected):
    mask = mask_section(np.array([wn_val]), K=K)
    assert mask[0] == expected

def test_mask_section_invalid_k():
    with pytest.raises(ValueError, match="K must be 1, 2, or 3"):
        mask_section(np.array([1000]), K=99)

# =====================================================================
# Unit Tests: R Interface (Mocked)
# =====================================================================

def test_smooth_spline_r_call_structure():
    """
    Test that we are constructing the R objects correctly.
    Mocking rpy2 to avoid needing R installed for this specific unit test.
    """
    with patch("spcproc.core.baseline.robjects") as mock_robj, \
         patch("spcproc.core.baseline._stats") as mock_stats:
        
        from spcproc.core.baseline import smooth_spline_r
        
        # Mock returns
        mock_stats.predict_smooth_spline.return_value = (None, [1, 2, 3]) # (x, y) tuple
        
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        w = np.array([1, 1, 1])
        
        smooth_spline_r(x, y, df=5, w=w)
        
        # Check if R vectors were created
        assert mock_robj.FloatVector.call_count >= 2
        # Check if stats::smooth.spline was called
        mock_stats.smooth_spline.assert_called_once()

# =====================================================================
# Integration Tests: BaselineCorrector
# =====================================================================

def test_baseline_corrector_validation(sample_spectra, config):
    """Test standard validation checks for bounds."""
    bc = BaselineCorrector(config)

    # 1. Zero WN out of range
    bc.config.zero_wn = 5000.0 # Max is 4000
    with pytest.raises(ValueError, match="Invalid zero-shift"):
        bc.run(sample_spectra)

    # 2. Stitch bounds out of range
    bc.config.zero_wn = 1380.0 # reset valid
    bc.config.stitch_bounds = (5000.0, 1340.0)
    with pytest.raises(ValueError, match="Invalid stitch_bounds"):
        bc.run(sample_spectra)

    # 3. Stitch bounds order wrong
    bc.config.stitch_bounds = (1340.0, 2070.0) # Low > High
    with pytest.raises(ValueError, match="expected order"):
        bc.run(sample_spectra)

def test_baseline_corrector_full_run(sample_spectra, config):
    """
    Smoke test running the full pipeline locally.
    Requires R environment to be present.
    """
    # Assuming R is installed, otherwise this needs a mark.skip
    bc = BaselineCorrector(config)
    results = bc.run(sample_spectra)

    required_keys = [
        "original_spectra", 
        "background_spectra", 
        "baselined_spectra", 
        "baselined_spectra_stitched_normalized"
    ]
    
    for key in required_keys:
        assert key in results
        df = results[key]
        assert "Wavenumber" in df.columns
        assert df.shape == sample_spectra.shape
        assert not df.isnull().values.any()

    # Check normalization logic specifically
    # Norm is calculated on wn > 1330. 
    # Since our synthetic data is roughly constant magnitude, the norm values should be reasonable.
    norm_df = results["baselined_spectra_stitched_normalized"]
    raw_df = results["baselined_spectra"]
    
    # Pick a point > 1330
    idx = norm_df["Wavenumber"] > 1330
    # Values in normed df should be scaled down compared to raw df
    # (assuming the norm factor > 1, which it usually is for absorbance)
    assert np.mean(norm_df.loc[idx, "Sample1"]) < np.mean(raw_df.loc[idx, "Sample1"])

def test_implicit_config_defaults(sample_spectra):
    """Test that the class handles missing config by loading defaults."""
    # Temporarily ensure defaults fit our dummy data range [400, 4000]
    # spcproc defaults are usually inside this range.
    bc = BaselineCorrector(config=None) 
    
    # Should not raise
    res = bc.run(sample_spectra)
    assert res is not None

def test_input_immutability(sample_spectra, config):
    """Ensure the input dataframe is not modified in place."""
    original_copy = sample_spectra.copy()
    bc = BaselineCorrector(config)
    
    _ = bc.run(sample_spectra)
    
    pd.testing.assert_frame_equal(sample_spectra, original_copy)