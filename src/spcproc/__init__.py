"""
spcproc: FTIR spectrum processing package

Core features:
- Baseline correction (OOP)
- Blank subtraction
- End-to-end pipeline

Typical usage:

    from spcproc import FTIRPipeline, PipelineConfig
"""

from .config import BaselineConfig, BlankConfig
from .core.baseline import BaselineCorrector
from .core.blank import BlankProcessor
from .core.pipeline import FTIRPipeline

__all__ = [
    "BaselineConfig",
    "BlankConfig",
    "PipelineConfig",
    "BaselineCorrector",
    "BlankProcessor",
    "FTIRPipeline",
]