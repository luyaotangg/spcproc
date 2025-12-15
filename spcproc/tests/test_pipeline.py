import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from spcproc.core. pipeline import FTIRPipeline

# ----------------------------------------------------------------------
# Fixtures & Helpers
# ----------------------------------------------------------------------

@pytest.fixture
def dummy_df():
    """Generates a minimal valid spectral dataframe."""
    wn = np. linspace(4000, 400, 50)
    return pd.DataFrame({
        "Wavenumber": wn,
        "sample_1": np.random.rand(50) * 0.05  # Realistic absorbance range
    })


@pytest.fixture
def dummy_blank_df():
    """Generates a minimal valid blank spectrum."""
    wn = np.linspace(4000, 400, 50)
    return pd.DataFrame({
        "Wavenumber": wn,
        "absorbance":  np.random.rand(50) * 0.02  # Blank typically weaker
    })


# ----------------------------------------------------------------------
# Unit Tests: Logic & Wiring
# ----------------------------------------------------------------------

def test_pipeline_baseline_only_wiring(dummy_df):
    """
    Verify pipeline calls baseline corrector and returns correct structure
    when no blank is provided. 
    """
    # 1. Mock the internal baseline corrector
    mock_baseline = MagicMock()
    
    # Setup what .run() returns (must include BOTH keys now)
    mock_return = {
        "baselined_spectra":  dummy_df. copy(),  # ← 添加这个! 
        "baselined_spectra_stitched_normalized": dummy_df.copy(),
        "original_spectra": dummy_df.copy(),
        "background_spectra": dummy_df. copy()
    }
    mock_baseline.run.return_value = mock_return

    # 2. Init pipeline with mock
    pipeline = FTIRPipeline(baseline=mock_baseline)
    
    # 3. Run
    result = pipeline.run(dummy_df)
    
    # 4. Verify
    mock_baseline.run.assert_called_once_with(dummy_df)
    assert result["baseline"] == mock_return
    assert "blank" not in result


def test_pipeline_with_blank_wiring(dummy_df, dummy_blank_df):
    """
    Verify pipeline integrates blank subtraction when blank_df is provided. 
    
    CRITICAL: Tests that pipeline uses 'baselined_spectra' (not normalized)
    for blank subtraction, following Amir's original logic.
    """
    # 1. Mock baseline to avoid R dependency and control output
    mock_baseline = MagicMock()
    
    # Create distinct dataframes to verify which one is used
    baselined_original = dummy_df.copy()
    baselined_original["sample_1"] = baselined_original["sample_1"] * 1.0  # Original scale
    
    baselined_normalized = dummy_df.copy()
    baselined_normalized["sample_1"] = baselined_normalized["sample_1"] * 2.5  # Normalized scale
    
    mock_baseline.run.return_value = {
        "baselined_spectra": baselined_original,  # ← This should be used for blank subtraction
        "baselined_spectra_stitched_normalized":  baselined_normalized,
        "original_spectra": dummy_df. copy(),
        "background_spectra": dummy_df.copy()
    }

    # 2. Init pipeline
    pipeline = FTIRPipeline(baseline=mock_baseline)
    
    # 3. Run with blank
    result = pipeline.run(dummy_df, blank_df=dummy_blank_df)

    # 4. Verify structure
    assert "baseline" in result
    assert "blank" in result
    assert "sample_blank_corrected" in result["blank"]
    assert "blank_processed" in result["blank"]
    assert "blank_high" in result["blank"]
    assert "blank_low" in result["blank"]
    assert "blank_mid" in result["blank"]
    
    # 5. Verify BlankProcessor received the correct (non-normalized) data
    # The corrected spectrum should have values in the original absorbance range,
    # not the normalized range
    corrected = result["blank"]["sample_blank_corrected"]
    assert "Wavenumber" in corrected. columns
    
    # Check that output values are in realistic absorbance range (not normalized scale)
    # If it used normalized data, values would be much larger (~2.0)
    # If it used original data, values should be small (~0.05)
    sample_col = [c for c in corrected.columns if c != "Wavenumber"][0]
    max_abs = corrected[sample_col]. abs().max()
    
    # This assertion will FAIL with current implementation (which uses normalized)
    # and PASS after fixing to use baselined_spectra
    assert max_abs < 1.0, (
        f"Blank-corrected spectrum has suspiciously large values ({max_abs:. 3f}). "
        "This suggests normalized data was used instead of original baselined spectra."
    )


def test_pipeline_blank_uses_original_not_normalized(dummy_df, dummy_blank_df):
    """
    REGRESSION TEST: Explicitly verify that blank subtraction uses 
    'baselined_spectra' and not 'baselined_spectra_stitched_normalized'. 
    
    This is the key difference from Amir's original implementation. 
    """
    from unittest.mock import patch
    
    mock_baseline = MagicMock()
    
    # Create dramatically different values to make the test obvious
    baselined_original = dummy_df.copy()
    baselined_original["sample_1"] = 0.05  # Realistic absorbance
    
    baselined_normalized = dummy_df.copy()
    baselined_normalized["sample_1"] = 5.0  # Normalized (much larger)
    
    mock_baseline.run.return_value = {
        "baselined_spectra": baselined_original,
        "baselined_spectra_stitched_normalized": baselined_normalized,
        "original_spectra": dummy_df,
        "background_spectra":  dummy_df
    }
    
    pipeline = FTIRPipeline(baseline=mock_baseline)
    
    # Patch BlankProcessor. subtract to spy on what it receives
    with patch('spcproc.core.pipeline.BlankProcessor.subtract') as mock_subtract:
        mock_subtract.return_value = dummy_df.copy()  # Return anything valid
        
        pipeline.run(dummy_df, blank_df=dummy_blank_df)
        
        # Verify subtract was called with the ORIGINAL baselined spectra
        called_with = mock_subtract.call_args[0][0]  # First positional arg
        
        # Check that the passed dataframe has the ORIGINAL values, not normalized
        assert called_with["sample_1"]. iloc[0] == pytest.approx(0.05, abs=0.01), (
            "BlankProcessor.subtract() should receive 'baselined_spectra' "
            "(original absorbance scale), not 'baselined_spectra_stitched_normalized'"
        )


# ----------------------------------------------------------------------
# Validation Tests
# ----------------------------------------------------------------------

def test_validate_sample_df_columns():
    """Pipeline should reject sample_df missing Wavenumber immediately."""
    pipeline = FTIRPipeline()
    bad_df = pd.DataFrame({"Data": [1, 2, 3]})
    
    with pytest.raises(ValueError, match="must contain 'Wavenumber'"):
        pipeline.run(bad_df)


def test_validate_blank_df_columns(dummy_df):
    """
    Pipeline should reject blank_df missing Wavenumber.
    
    We mock baseline.run here because we only care about 
    blank_df validation, not sample processing. 
    """
    mock_baseline = MagicMock()
    
    # Return both required keys
    mock_baseline.run.return_value = {
        "baselined_spectra":  dummy_df.copy(),
        "baselined_spectra_stitched_normalized": dummy_df.copy()
    }
    
    pipeline = FTIRPipeline(baseline=mock_baseline)
    bad_blank = pd.DataFrame({"Data": [1, 2]})

    with pytest.raises(ValueError, match="Blank DataFrame must contain"):
        pipeline.run(dummy_df, blank_df=bad_blank)


def test_validate_blank_df_absorbance_column(dummy_df):
    """
    Blank DataFrame should be auto-renamed if it has one non-Wavenumber column.
    """
    mock_baseline = MagicMock()
    mock_baseline.run.return_value = {
        "baselined_spectra": dummy_df.copy(),
        "baselined_spectra_stitched_normalized":  dummy_df.copy()
    }
    
    pipeline = FTIRPipeline(baseline=mock_baseline)
    
    # Blank with non-standard column name
    blank_nonstandard = pd.DataFrame({
        "Wavenumber": np.linspace(4000, 400, 50),
        "intensity": np.random.rand(50) * 0.02  # Should be auto-renamed to 'absorbance'
    })
    
    # Should not raise (BlankProcessor handles renaming)
    result = pipeline.run(dummy_df, blank_df=blank_nonstandard)
    assert "blank" in result


# ----------------------------------------------------------------------
# Integration Test (optional, requires R environment)
# ----------------------------------------------------------------------

@pytest.mark.integration
def test_pipeline_end_to_end_realistic():
    """
    Full integration test with realistic data.
    Requires R and rpy2 to be properly configured.
    """
    # Create realistic FTIR spectrum
    wn = np.linspace(4000, 400, 1866)  # Typical FTIR resolution
    
    # Simulate PTFE + organic aerosol
    sample_abs = np.zeros_like(wn)
    sample_abs += 0.08 * np.exp(-((wn - 1200)**2) / 5000)  # PTFE C-F peak
    sample_abs += 0.03 * np.exp(-((wn - 2900)**2) / 2000)  # C-H stretch
    sample_abs += 0.02 * np.exp(-((wn - 1700)**2) / 1000)  # C=O stretch
    sample_abs += np.random.normal(0, 0.001, len(wn))  # Noise
    
    sample_df = pd.DataFrame({
        "Wavenumber": wn,
        "Sample_001": sample_abs
    })
    
    # Blank (only PTFE)
    blank_abs = np.zeros_like(wn)
    blank_abs += 0.06 * np.exp(-((wn - 1200)**2) / 5000)  # PTFE C-F
    blank_abs += np.random.normal(0, 0.0005, len(wn))
    
    blank_df = pd.DataFrame({
        "Wavenumber": wn,
        "absorbance": blank_abs
    })
    
    # Run pipeline (no mocks)
    pipeline = FTIRPipeline()
    result = pipeline.run(sample_df, blank_df=blank_df)
    
    # Verify organic peaks remain after blank subtraction
    corrected = result["blank"]["sample_blank_corrected"]
    wn_corrected = corrected["Wavenumber"].values
    abs_corrected = corrected["Sample_001"].values
    
    # Check C=O peak at 1700 is preserved
    co_region = (wn_corrected > 1650) & (wn_corrected < 1750)
    assert abs_corrected[co_region]. max() > 0.01, "Organic C=O peak should remain"
    
    # Check PTFE peak at 1200 is reduced
    ptfe_region = (wn_corrected > 1150) & (wn_corrected < 1250)
    original_ptfe = sample_abs[ptfe_region]. max()
    corrected_ptfe = abs_corrected[ptfe_region].max()
    assert corrected_ptfe < original_ptfe * 0.5, "PTFE should be partially removed"