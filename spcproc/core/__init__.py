"""
Core modules for FTIR processing.
"""

from .baseline import BaselineCorrector
from .blank import BlankProcessor
from .pipeline import FTIRPipeline

__all__ = ["BaselineCorrector", "BlankProcessor", "FTIRPipeline"]