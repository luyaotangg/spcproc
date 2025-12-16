"""
Minimal working example for spcproc.

This script demonstrates the user-facing workflow:
1) Load example input data (sample + blank) from data/examples/
2) Run the full FTIR pipeline (baseline correction + blank subtraction)
3) Save the final corrected spectra to an output CSV file

Intended audience:
- Users with limited programming experience
- Quick sanity check after installation

Usage:
    python scripts/run_examples.py
"""

from pathlib import Path

from spcproc.io import load_from_files
from spcproc.core.pipeline import FTIRPipeline


def main():
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "examples"
    out_dir = project_root / "data" / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = data_dir / "sample.csv"
    blank_path = data_dir / "blank.csv"
    output_path = out_dir / "output_example.csv"

    # Load example data
    sample_df, blank_df = load_from_files(
        sample_path=sample_path,
        blank_path=blank_path,
    )

    # Run pipeline
    pipeline = FTIRPipeline()
    out = pipeline.run(sample_df=sample_df, blank_df=blank_df)

    # Save final result
    final = out["blank"]["sample_blank_corrected"]
    final.to_csv(output_path, index=False)

    print("Example pipeline finished successfully.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()