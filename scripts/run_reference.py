"""
Reference reproduction / diagnostic script for spcproc.

This script runs the pipeline on Amir's PSI reference dataset in data/reference/
and exports intermediate processing steps (CSV + plots) for validation and debugging.

Intended audience:
- Developers / maintainers
- Method validation against the original R workflow
- Generating figures and step-by-step outputs for the report/appendix

Usage:
    python scripts/run_reference.py
"""


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from spcproc.core.baseline import BaselineCorrector
from spcproc.core.blank import BlankProcessor
from spcproc.config import BaselineConfig, BlankConfig


DATA_DIR = Path("data")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "reference" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)  # Output path

OUT_DIR.mkdir(exist_ok=True, parents=True)


def plot_step(
    df:  pd.DataFrame,
    outdir: Path,
    filename: str,
    title: str,
    cols=None,
    xlim=(4000, 400),
    ylim=None,
    y_label:  str = "Absorbance",
):
    if "Wavenumber" not in df.columns:
        raise ValueError("DataFrame must contain 'Wavenumber' column.")

    outdir.mkdir(exist_ok=True, parents=True)
    savepath = outdir / filename

    if cols is None:
        all_cols = [c for c in df.columns if c != "Wavenumber"]
        cols = all_cols[:  min(5, len(all_cols))]
    if not cols:
        return

    plt.figure(figsize=(7, 4))
    x = df["Wavenumber"]. values

    for col in cols:
        y = df[col].values
        plt.plot(x, y, label=col, linewidth=0.7, alpha=0.8)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.gca().invert_xaxis()  # 4000 → 400

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt. ylim(*ylim)

    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()



def main():
    scenario_name = "orig"
    scenario_dir = OUT_DIR / scenario_name
    scenario_dir.mkdir(exist_ok=True, parents=True)

    psi_raw = pd.read_csv(DATA_DIR / "reference/FTIR_raw_spectra.csv")

    # Step 01: raw PSI spectrum
    psi_raw.to_csv(
        scenario_dir / f"step01_raw_PSI_spectra_{scenario_name}.csv",
        index=False,
    )
    plot_step(
        df=psi_raw,
        outdir=scenario_dir,
        filename=f"step01_raw_PSI_spectra_{scenario_name}.png",
        title=f"Step 01 ({scenario_name}): Raw spectra",
    )

    base_cfg = BaselineConfig()          # Use default zero_wn and stitch_bounds
    bc = BaselineCorrector(config=base_cfg)
    base_result = bc.run(psi_raw)

    zero_shifted = base_result["original_spectra"]
    background = base_result["background_spectra"]
    baselined = base_result["baselined_spectra"]

    # Step 02–04：Write CSV and plot
    zero_shifted.to_csv(
        scenario_dir / f"step02_zero_shifted_spectra_{scenario_name}.csv",
        index=False,
    )
    background.to_csv(
        scenario_dir / f"step03_background_spectra_stitched_{scenario_name}.csv",
        index=False,
    )
    baselined.to_csv(
        scenario_dir / f"step04_baselined_spectra_stitched_{scenario_name}.csv",
        index=False,
    )

    plot_step(
        df=zero_shifted,
        outdir=scenario_dir,
        filename=f"step02_zero_shifted_spectra_{scenario_name}.png",
        title=f"Step 02 ({scenario_name}): Zero-shifted spectra",
    )
    plot_step(
        df=background,
        outdir=scenario_dir,
        filename=f"step03_background_spectra_stitched_{scenario_name}.png",
        title=f"Step 03 ({scenario_name}): Estimated background (stitched)",
    )
    plot_step(
        df=baselined,
        outdir=scenario_dir,
        filename=f"step04_baselined_spectra_stitched_{scenario_name}.png",
        title=f"Step 04 ({scenario_name}): Baselined spectra (stitched)",
        ylim=(-0.025, 0.08),
    )

    # Blank processing starts here
    blank_raw = pd.read_csv(
        DATA_DIR / "reference/Blank_baselinecorrected_filter_spectrum.csv"
    )

    blank_cfg = BlankConfig()
    bp = BlankProcessor(blank_df=blank_raw, config=blank_cfg)

    blank_proc = bp.blank_processed
    high = bp.high_blank
    low = bp. low_blank
    mid = bp.mid_blank

    # Step 05–08：Write CSV and plot
    blank_proc.to_csv(
        scenario_dir / f"step05_blank_processed_{scenario_name}.csv",
        index=False,
    )
    high.to_csv(
        scenario_dir / f"step06_blank_region_high_{scenario_name}.csv",
        index=False,
    )
    low.to_csv(
        scenario_dir / f"step07_blank_region_low_{scenario_name}.csv",
        index=False,
    )
    mid.to_csv(
        scenario_dir / f"step08_blank_region_middle_{scenario_name}.csv",
        index=False,
    )

    plot_step(
        df=blank_proc,
        outdir=scenario_dir,
        filename=f"step05_blank_processed_{scenario_name}.png",
        title=f"Step 05 ({scenario_name}): Processed blank spectrum",
        cols=["absorbance"],
        y_label="Absorbance",
    )
    plot_step(
        df=high,
        outdir=scenario_dir,
        filename=f"step06_blank_region_high_{scenario_name}.png",
        title=f"Step 06 ({scenario_name}): Blank – high region (>2500 cm$^{{-1}}$)",
        cols=["absorbance"],
        y_label="Absorbance",
    )
    plot_step(
        df=low,
        outdir=scenario_dir,
        filename=f"step07_blank_region_low_{scenario_name}.png",
        title=f"Step 07 ({scenario_name}): Blank – low region (<=810 cm$^{{-1}}$)",
        cols=["absorbance"],
        y_label="Absorbance",
    )
    plot_step(
        df=mid,
        outdir=scenario_dir,
        filename=f"step08_blank_region_middle_{scenario_name}.png",
        title=f"Step 08 ({scenario_name}): Blank – middle region (810–2500 cm$^{{-1}}$)",
        cols=["absorbance"],
        y_label="Absorbance",
    )

    # Step 09：blank subtraction
    final = bp.subtract(baselined)

    final.to_csv(
        scenario_dir
        / f"step09_final_baselined_spectra_after_blanksub_{scenario_name}.csv",
        index=False,
    )

    plot_step(
        df=final,
        outdir=scenario_dir,
        filename=f"step09_final_baselined_after_blanksub_{scenario_name}.png",
        title=f"Step 09 ({scenario_name}): Final spectra after blank subtraction",
        ylim=(-0.01, 0.06),
    )

    print("All steps finished. Results saved under:", scenario_dir)


if __name__ == "__main__":
    main()