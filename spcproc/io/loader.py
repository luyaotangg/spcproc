from pathlib import Path
from typing import Union
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

PathLike = Union[str, Path]


def _ensure_wavenumber(df: pd.DataFrame, source=None) -> pd.DataFrame:
    if "Wavenumber" not in df.columns:
        src = f" ({source})" if source else ""
        raise ValueError(f"DataFrame{src} must contain a 'Wavenumber' column.")
    return df


def load_sample_csv(path: PathLike, use_data_dir: bool = False) -> pd.DataFrame:
    """Load and validate sample spectra from CSV."""
    fpath = DATA_DIR / path if use_data_dir else Path(path)
    return load_sample_df(pd.read_csv(fpath), source=fpath)


def load_sample_df(df: pd.DataFrame, source=None) -> pd.DataFrame:
    return _ensure_wavenumber(df, source=source)


def load_blank_csv(path: PathLike, use_data_dir: bool = False) -> pd.DataFrame:
    """Load and validate blank spectrum CSV."""
    fpath = DATA_DIR / path if use_data_dir else Path(path)
    return load_blank_df(pd.read_csv(fpath), source=fpath)


def load_blank_df(df: pd.DataFrame, source=None) -> pd.DataFrame:
    """
    Validate blank dataframe. 
    Auto-renames single intensity column to 'absorbance' if needed.
    """
    df = _ensure_wavenumber(df, source).copy()

    if "absorbance" not in df.columns:
        other_cols = [c for c in df.columns if c != "Wavenumber"]
        
        if len(other_cols) == 1:
            df.rename(columns={other_cols[0]: "absorbance"}, inplace=True)
        else:
            src = f" ({source})" if source else ""
            raise ValueError(
                f"Blank DataFrame{src} must contain an 'absorbance' column, "
                "or exactly one non-'Wavenumber' column."
            )
    return df


def save_spectra_csv(df: pd.DataFrame, filename: str, use_data_dir: bool = False) -> None:
    path = DATA_DIR / filename if use_data_dir else Path(filename)
    df.to_csv(path, index=False)