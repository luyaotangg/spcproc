# repo/run_steps_oop.py

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from spcproc.core.baseline import BaselineCorrector
from spcproc.core.blank import BlankProcessor
from spcproc.config import BaselineConfig, BlankConfig


DATA_DIR = Path("data")
OUT_DIR = Path("py_steps")   # 输出目录（新的）

OUT_DIR.mkdir(exist_ok=True, parents=True)


# -------------------------------------------------------------------
# 通用画图函数：基本沿用你之前的 plot_step
# -------------------------------------------------------------------
def plot_step(
    df: pd.DataFrame,
    outdir: Path,
    filename: str,
    title: str,
    cols=None,
    xlim=(4000, 400),
    ylim=None,
    y_label: str = "Absorbance",
):
    """
    df: 含 'Wavenumber' 的 DataFrame
    outdir: 输出目录（Path）
    filename: 保存文件名（字符串）
    cols: 要画的样本列（不含 Wavenumber），默认前 5 条
    """
    if "Wavenumber" not in df.columns:
        raise ValueError("DataFrame must contain 'Wavenumber' column.")

    outdir.mkdir(exist_ok=True, parents=True)
    savepath = outdir / filename

    # 选择要画的列
    if cols is None:
        all_cols = [c for c in df.columns if c != "Wavenumber"]
        cols = all_cols[: min(5, len(all_cols))]
    if not cols:
        return

    plt.figure(figsize=(7, 4))
    x = df["Wavenumber"].values

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
        plt.ylim(*ylim)

    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


# -------------------------------------------------------------------
# 主流程：只跑一个场景（orig, 全分辨率），但导出 step01–step11
# -------------------------------------------------------------------
def main():
    scenario_name = "orig"
    scenario_dir = OUT_DIR / scenario_name
    scenario_dir.mkdir(exist_ok=True, parents=True)

    # ---------- 读原始 PSI 光谱 ----------
    psi_raw = pd.read_csv(DATA_DIR / "FTIR_raw_spectra.csv")

    # Step 01: 原始 PSI 光谱
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

    # ---------- Baseline 部分：使用新的 BaselineCorrector ----------
    base_cfg = BaselineConfig()          # 使用默认 zero_wn / stitch_bounds
    bc = BaselineCorrector(config=base_cfg)
    base_result = bc.run(psi_raw)

    zero_shifted = base_result["original_spectra"]
    background = base_result["background_spectra"]
    baselined = base_result["baselined_spectra"]
    baselined_norm = base_result["baselined_spectra_stitched_normalized"]

    # Step 02–05：写 CSV + 画图（文件名尽量和旧脚本对齐）
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
    baselined_norm.to_csv(
        scenario_dir / f"step05_baselined_spectra_normalized_{scenario_name}.csv",
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
        title=f"Step 04 ({scenario_name}): Baselined spectra (stitched, not normalized)",
        ylim=(-0.025, 0.08),
    )
    plot_step(
        df=baselined_norm,
        outdir=scenario_dir,
        filename=f"step05_baselined_spectra_normalized_{scenario_name}.png",
        title=f"Step 05 ({scenario_name}): Baselined + 2-norm normalized spectra",
        ylim=(-0.5, 2),
    )

    # ---------- Blank 部分：使用新的 BlankProcessor ----------
    blank_raw = pd.read_csv(
        DATA_DIR / "Blank_baselinecorrected_filter_spectrum.csv"
    )

    blank_cfg = BlankConfig()
    bp = BlankProcessor(blank_df=blank_raw, config=blank_cfg)

    blank_proc = bp.blank_processed
    high = bp.high_blank
    low = bp.low_blank
    mid = bp.mid_blank

    # Step 07–10：写 CSV + 图
    blank_proc.to_csv(
        scenario_dir / f"step07_blank_processed_{scenario_name}.csv",
        index=False,
    )
    high.to_csv(
        scenario_dir / f"step08_blank_region_high_{scenario_name}.csv",
        index=False,
    )
    low.to_csv(
        scenario_dir / f"step09_blank_region_low_{scenario_name}.csv",
        index=False,
    )
    mid.to_csv(
        scenario_dir / f"step10_blank_region_middle_{scenario_name}.csv",
        index=False,
    )

    plot_step(
        df=blank_proc,
        outdir=scenario_dir,
        filename=f"step07_blank_processed_{scenario_name}.png",
        title=f"Step 07 ({scenario_name}): Processed blank spectrum",
        cols=["absorbance"],
        y_label="Absorbance",
    )
    plot_step(
        df=high,
        outdir=scenario_dir,
        filename=f"step08_blank_region_high_{scenario_name}.png",
        title=f"Step 08 ({scenario_name}): Blank – high region (>2500 cm$^{{-1}}$)",
        cols=["absorbance"],
        y_label="Absorbance",
    )
    plot_step(
        df=low,
        outdir=scenario_dir,
        filename=f"step09_blank_region_low_{scenario_name}.png",
        title=f"Step 09 ({scenario_name}): Blank – low region (<=810 cm$^{{-1}}$)",
        cols=["absorbance"],
        y_label="Absorbance",
    )
    plot_step(
        df=mid,
        outdir=scenario_dir,
        filename=f"step10_blank_region_middle_{scenario_name}.png",
        title=f"Step 10 ({scenario_name}): Blank – middle region (810–2500 cm$^{{-1}}$)",
        cols=["absorbance"],
        y_label="Absorbance",
    )

    # ---------- Step 11：blank subtraction（和你原来一样，用未 normalized 的 baselined_spectra） ----------
    final = bp.subtract(baselined)

    final.to_csv(
        scenario_dir
        / f"step11_final_baselined_spectra_after_blanksub_{scenario_name}.csv",
        index=False,
    )

    plot_step(
        df=final,
        outdir=scenario_dir,
        filename=f"step11_final_baselined_after_blanksub_{scenario_name}.png",
        title=f"Step 11 ({scenario_name}): Final spectra after blank subtraction",
        ylim=(-0.01, 0.06),
    )

    print("All steps finished. Results saved under:", scenario_dir)


if __name__ == "__main__":
    main()