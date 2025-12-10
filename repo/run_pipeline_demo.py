# run_pipeline_demo.py

import matplotlib.pyplot as plt

from spcproc.core.pipeline import FTIRPipeline
from spcproc.io import (
    load_sample_csv,
    load_blank_csv,
    save_spectra_csv,
)

def main():
    # ====== 1. 读入数据（根据你自己的文件名改） ======
    # 如果用的是 io/loader 里默认的 DATA_DIR（spcproc 上上层的 data/ 文件夹）
    sample_df = load_sample_csv("FTIR_raw_spectra.csv", use_data_dir=True)
    blank_df  = load_blank_csv("Blank_baselinecorrected_filter_spectrum.csv", use_data_dir=True)

    # 如果你更想给绝对路径 / 相对路径，也可以：
    # sample_df = load_sample_csv("/path/to/your/sample.csv", use_data_dir=False)
    # blank_df  = load_blank_csv("/path/to/your/blank.csv", use_data_dir=False)

    print("Sample columns:", sample_df.columns)
    print("Blank columns:", blank_df.columns)

    # ====== 2. 建一个 pipeline，用默认 config 跑一遍 ======
    pipe = FTIRPipeline()
    result = pipe.run(sample_df=sample_df, blank_df=blank_df)

    # 看一下返回的 key
    print("Top-level keys:", result.keys())
    print("Baseline keys:", result["baseline"].keys())
    print("Blank keys:", result["blank"].keys())

    # 取出两个最关键的结果
    baselined_norm = result["baseline"]["baselined_spectra_stitched_normalized"]
    blank_corrected = result["blank"]["sample_blank_corrected"]

    # ====== 3. 简单画一下前几条光谱对比（肉眼 sanity check） ======
    wn = baselined_norm["Wavenumber"].values
    sample_cols = [c for c in baselined_norm.columns if c != "Wavenumber"]

    # 只看前 3 条，避免太乱
    cols_to_plot = sample_cols[:3]

    plt.figure(figsize=(8, 4))
    for col in cols_to_plot:
        plt.plot(wn, baselined_norm[col], label=f"Baseline-only: {col}", alpha=0.5)
    for col in cols_to_plot:
        plt.plot(wn, blank_corrected[col], linestyle="--", label=f"Blank-corrected: {col}")
    plt.gca().invert_xaxis()  # FTIR 习惯反过来
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Absorbance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ====== 4. 把结果存成 CSV 方便后面用 ======
    save_spectra_csv(baselined_norm, "baselined_normalized.csv", use_data_dir=True)
    save_spectra_csv(blank_corrected, "blank_subtracted.csv", use_data_dir=True)
    print("Saved baselined_normalized.csv and blank_subtracted.csv to data/")

if __name__ == "__main__":
    main()