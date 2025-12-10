import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from spcproc.core.pipeline import FTIRPipeline

# ----------------------------------------------------------------------
# Fixtures & Helpers
# ----------------------------------------------------------------------

@pytest.fixture
def dummy_df():
    """Generates a minimal valid spectral dataframe."""
    wn = np.linspace(4000, 400, 50)
    return pd.DataFrame({
        "Wavenumber": wn,
        "sample_1": np.random.rand(50)
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
    # Setup what .run() returns
    mock_return = {
        "baselined_spectra_stitched_normalized": dummy_df, # key needed for blank step
        "original_spectra": dummy_df
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


def test_pipeline_with_blank_wiring(dummy_df):
    """
    Verify pipeline integrates blank subtraction when blank_df is provided.
    """
    # 1. Mock baseline (to avoid R errors and speed up test)
    mock_baseline = MagicMock()
    mock_baseline.run.return_value = {
        "baselined_spectra_stitched_normalized": dummy_df
    }

    # 2. Init pipeline
    pipeline = FTIRPipeline(baseline=mock_baseline)
    
    # 3. Run with blank
    # We use a valid blank_df so BlankProcessor doesn't crash on init
    blank_df = dummy_df.copy()
    result = pipeline.run(dummy_df, blank_df=blank_df)

    # 4. Verify
    assert "blank" in result
    assert "sample_blank_corrected" in result["blank"]
    # Ensure BlankProcessor logic was actually triggered (output keys exist)
    assert result["blank"]["blank_processed"] is not None


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
    
    CRITICAL: We mock baseline.run here because we don't care about 
    sample processing success, we only care that it fails LATER 
    when checking blank_df.
    """
    mock_baseline = MagicMock()
    # Return a dummy result so Step 1 passes
    mock_baseline.run.return_value = {
        "baselined_spectra_stitched_normalized": dummy_df
    }
    
    pipeline = FTIRPipeline(baseline=mock_baseline)
    bad_blank = pd.DataFrame({"Data": [1, 2]})

    with pytest.raises(ValueError, match="Blank DataFrame must contain"):
        pipeline.run(dummy_df, blank_df=bad_blank)