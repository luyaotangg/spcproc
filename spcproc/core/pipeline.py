from typing import Dict, Optional
import pandas as pd

from spcproc.core.baseline import BaselineCorrector
from spcproc.config import BaselineConfig, BlankConfig
from spcproc.core.blank import BlankProcessor


class FTIRPipeline:
    """
    Orchestrates the full FTIR processing workflow:
    Baseline Correction -> (Optional) Blank Subtraction.
    """

    def __init__(
        self,
        baseline: Optional[BaselineCorrector] = None,
        blank_config: Optional[BlankConfig] = None,
    ):
        self.baseline = baseline or BaselineCorrector(BaselineConfig())
        self.blank_config = blank_config or BlankConfig()

    def run(
        self,
        sample_df: pd.DataFrame,
        blank_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        
        if "Wavenumber" not in sample_df.columns:
            raise ValueError("Input DataFrame must contain 'Wavenumber' column.")

        # 1. Baseline correction
        base_result = self.baseline.run(sample_df)
        baselined_norm = base_result["baselined_spectra_stitched_normalized"]

        output = {"baseline": base_result}

        # 2. Blank subtraction (if blank provided)
        if blank_df is not None:
            if "Wavenumber" not in blank_df.columns:
                raise ValueError("Blank DataFrame must contain 'Wavenumber' column.")

            blank_processor = BlankProcessor(
                blank_df=blank_df,
                config=self.blank_config,
            )

            blank_corrected = blank_processor.subtract(baselined_norm)

            output["blank"] = {
                "blank_processed": blank_processor.blank_processed,
                "blank_high": blank_processor.high_blank,
                "blank_low": blank_processor.low_blank,
                "blank_mid": blank_processor.mid_blank,
                "sample_blank_corrected": blank_corrected,
            }

        return output