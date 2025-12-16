# spcproc — FTIR Baseline & Blank Correction

A lightweight, modular Python implementation of the FTIR baseline-correction and
blank-subtraction workflow originally developed by **Amir** for PSI
Teflon-filter FTIR spectra.

This package reproduces the essential logic of the original R script while
providing a clearer, maintainable, and extensible Python structure focused on
usability and reproducibility.

---

## Features

- **Baseline correction** using region-wise smoothing spline
  (`smooth.spline` via `rpy2`)
- **High / mid / low spectral stitching**
- **Zero-shift alignment** at a fixed reference wavenumber
- **Blank preprocessing & subtraction** with masking and scaling windows
- **Unified processing interface** via `FTIRPipeline.run(sample_df, blank_df)`

---

## Input Data Format

Both sample and blank spectra must be provided as CSV files with the following
structure:

- The first column must be named `Wavenumber` (unit: cm⁻¹).
- Sample files may contain one or more spectra columns.
- Blank files must contain exactly one absorbance column.
- The wavenumber range must cover the reference region around **1336.5996 cm⁻¹**
  required for zero-shift alignment.

The `load_from_files()` function validates the input format and provides
informative error messages if the requirements are not met.

The files in `data/examples/` provide minimal example inputs illustrating the
expected data format. They are synthetic and intended for demonstration and
testing purposes only, not to represent real measurements.

---

## Quick Example (Python API)

```python
from spcproc.io import load_from_files
from spcproc.core.pipeline import FTIRPipeline

# Load sample and blank spectra from CSV files
sample_df, blank_df = load_from_files(
    sample_path="data/examples/sample.csv",
    blank_path="data/examples/blank.csv",
)

# Initialize and run pipeline
pipeline = FTIRPipeline()
out = pipeline.run(sample_df=sample_df, blank_df=blank_df)

# Access final blank-corrected spectra
final = out["blank"]["sample_blank_corrected"]
print(final.head())
```
## Run the Example Script
A minimal working example using the files in data/examples/ is provided as a
script:
python scripts/run_examples.py

The script runs the full pipeline and writes the output to:
data/examples/output/output_example.csv

## Default Reference Parameters

This library is designed for **PSI Teflon-filter FTIR spectra**, which use a
**fixed instrument sampling grid**. Therefore, the reference wavenumbers used
for:

- Zero-shift alignment  
- Region boundaries (high → mid → low)

are constant across all datasets processed with this library.

Default values (defined in `config.py`):

- **Zero-shift anchor:** `1336.5996 cm⁻¹`
- **High–mid boundary:** `2069.5114 cm⁻¹`
- **Mid–low boundary:** `1336.5996 cm⁻¹`

Their scientific origin and recomputation utility are documented in:

```text
spcproc/core/reference.py
```
These values were calculated from Amir’s original PSI FTIR datasets.

## Reference Datasets (Zenodo)

The reference parameters used in this library were obtained from Amir’s publicly
available PSI FTIR dataset:

- https://zenodo.org/records/4882967

This dataset defines the instrument’s wavenumber sampling grid and underpins the
reference anchors used throughout the pipeline.

## Project Structure

```text
SIE_PROJECT/
├── src/
│   └── spcproc/                # Main Python package
│       ├── core/               # Core algorithmic logic
│       ├── io/                 # Data loading & user-facing I/O utilities
│       ├── config.py           # Default parameters
│       └── __init__.py
│
├── tests/                      # Unit tests for reproducibility
├── data/
│   ├── examples/               # Minimal example input data
│   └── reference/              # Reference datasets from Amir (PSI)
│
├── scripts/
│   ├── run_examples.py         # Minimal working example
│   └── run_reference.py        # Diagnostic / step-by-step reproduction
│
├── pyproject.toml              # Package configuration
├── requirements.txt
└── README.md
```

## Folder Notes

- **spcproc/**: Core library implementing the FTIR baseline-correction and
  blank-subtraction algorithms.
- **tests/**: Pytest suite ensuring numerical consistency with the original
  R scripts.
- **data/examples/**: Minimal example input files for quick testing.
- **data/reference/**: Reference datasets from Amir used for method validation.
- **scripts/**: Executable scripts for running the example workflow and
  reproducing the reference processing steps.

---

## Scope and Limitations

This implementation is specifically tuned for PSI Teflon-filter FTIR spectra
with a fixed sampling grid. It is not intended as a general-purpose baseline
correction algorithm for arbitrary FTIR datasets.