# spcproc — FTIR Baseline & Blank Correction

A lightweight, modular Python implementation of the FTIR baseline-correction and blank-subtraction workflow originally developed by **Amir** for PSI Teflon-filter FTIR spectra.

This package reproduces the essential logic of the original R script while providing a clearer, maintainable Python structure.

## Features

* **Baseline correction** using region-wise smoothing spline (`smooth.spline` via rpy2)
* **High / mid / low spectral stitching**
* **Zero-shift alignment** at a fixed reference wavenumber
* **Blank preprocessing & subtraction** with masking and scaling windows
* **Unified processing interface** via `FTIRPipeline.run(sample_df, blank_df)`

## Default Reference Parameters

This library is designed for **PSI Teflon-filter FTIR spectra**, which use a **fixed instrument sampling grid**. Therefore, the reference wavenumbers used for:

* Zero-shift alignment
* Region boundaries (high → mid → low)

are **constant across all datasets** processed with this library.

Default values (defined in `config.py`):

* **Zero-shift anchor:** `1336.5996 cm⁻¹`
* **High–mid boundary:** `2069.5114 cm⁻¹`
* **Mid–low boundary:** `1336.5996 cm⁻¹`

Their scientific origin and recomputation utility are documented in:

`spcproc/core/reference.py`

These values were calculated from Amir’s original PSI FTIR datasets (see below).

## Reference Datasets (Zenodo)

The reference parameters used in this library were obtained from Amir’s publicly available PSI FTIR dataset:

* [https://zenodo.org/records/4882967](https://zenodo.org/records/4882967)

This dataset defines the instrument’s wavenumber sampling grid, and therefore the anchor wavenumbers used throughout this pipeline.

## Quick Example

```python
import pandas as pd
from spcproc.core.pipeline import FTIRPipeline

# Load data (expecting 'Wavenumber' column)
sample = pd.read_csv("data/sample.csv")
blank  = pd.read_csv("data/blank.csv")

# Initialize and run
pipeline = FTIRPipeline()
out = pipeline.run(sample_df=sample, blank_df=blank)

# Access results
# 'baseline' dict contains intermediate baseline correction steps
# 'blank' dict contains final blank-subtracted results
baselined = out["baseline"]["baselined_spectra_stitched_normalized"]
final = out["blank"]["sample_blank_corrected"]

print(final.head())
```

## Project Structure

```text
SIE_PROJECT/
├── spcproc/                  # Main Python package
│   ├── core/                 # Core logic (baseline, blank, reference)
│   ├── config.py             # Default parameters
│   └── tests/                # Unit tests for reproducibility
├── data/                     # Example raw and blank spectra
├── repo/                     # Execution scripts (run pipeline & save steps)
└── README.md
```

### Folder Notes

* **spcproc/**: The actual library source code.
* **tests/**: Pytest suite ensuring numerical consistency with original R scripts.
* **repo/**: Contains an execution script that runs the pipeline on data/ and export intermediate CSVs and plots.