import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_BINARY_COLUMNS = ["id", "binary_pred", "phq9_pred"]
REQUIRED_TERNARY_COLUMNS = ["id", "ternary_pred", "phq9_pred"]


def read_csv(csv_path: str) -> pd.DataFrame:
    """Read a CSV file with common encodings."""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            last_error = e

    raise ValueError(f"Failed to read {csv_path}. Last error: {last_error}")


def validate_columns(df: pd.DataFrame, required_columns: list[str], file_name: str) -> None:
    """Check whether required columns exist."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {missing}")


def validate_no_duplicate_ids(df: pd.DataFrame, file_name: str) -> None:
    """Check duplicate sample IDs."""
    duplicated = df[df["id"].duplicated()]["id"].tolist()
    if duplicated:
        raise ValueError(f"{file_name} contains duplicated IDs, for example: {duplicated[:10]}")


def validate_id_consistency(binary_df: pd.DataFrame, ternary_df: pd.DataFrame) -> None:
    """Check whether binary.csv and ternary.csv contain the same sample IDs."""
    binary_ids = set(binary_df["id"].astype(str).tolist())
    ternary_ids = set(ternary_df["id"].astype(str).tolist())

    missing_in_ternary = sorted(binary_ids - ternary_ids)
    missing_in_binary = sorted(ternary_ids - binary_ids)

    if missing_in_ternary:
        raise ValueError(f"ternary.csv is missing IDs that exist in binary.csv: {missing_in_ternary[:10]}")
    if missing_in_binary:
        raise ValueError(f"binary.csv is missing IDs that exist in ternary.csv: {missing_in_binary[:10]}")


def validate_binary_predictions(binary_df: pd.DataFrame) -> None:
    """Validate binary predictions and PHQ-9 predictions."""
    for col in ["binary_pred", "phq9_pred"]:
        binary_df[col] = pd.to_numeric(binary_df[col], errors="coerce")

    if binary_df["binary_pred"].isna().any():
        raise ValueError("binary.csv contains empty or non-numeric binary_pred values.")

    if binary_df["phq9_pred"].isna().any():
        raise ValueError("binary.csv contains empty or non-numeric phq9_pred values.")

    binary_values = set(binary_df["binary_pred"].astype(int).tolist())
    if not binary_values.issubset({0, 1}):
        raise ValueError("binary_pred must only contain 0 or 1.")

    if not np.isfinite(binary_df["phq9_pred"]).all():
        raise ValueError("binary.csv contains non-finite phq9_pred values.")

    out_of_range = binary_df[(binary_df["phq9_pred"] < 0) | (binary_df["phq9_pred"] > 27)]
    if not out_of_range.empty:
        preview = out_of_range[["id", "phq9_pred"]].head(10).to_dict(orient="records")
        raise ValueError(f"binary.csv phq9_pred must be in [0, 27]. Examples: {preview}")


def validate_ternary_predictions(ternary_df: pd.DataFrame) -> None:
    """Validate ternary predictions and PHQ-9 predictions."""
    for col in ["ternary_pred", "phq9_pred"]:
        ternary_df[col] = pd.to_numeric(ternary_df[col], errors="coerce")

    if ternary_df["ternary_pred"].isna().any():
        raise ValueError("ternary.csv contains empty or non-numeric ternary_pred values.")

    if ternary_df["phq9_pred"].isna().any():
        raise ValueError("ternary.csv contains empty or non-numeric phq9_pred values.")

    ternary_values = set(ternary_df["ternary_pred"].astype(int).tolist())
    if not ternary_values.issubset({0, 1, 2}):
        raise ValueError("ternary_pred must only contain 0, 1, or 2.")

    if not np.isfinite(ternary_df["phq9_pred"]).all():
        raise ValueError("ternary.csv contains non-finite phq9_pred values.")

    out_of_range = ternary_df[(ternary_df["phq9_pred"] < 0) | (ternary_df["phq9_pred"] > 27)]
    if not out_of_range.empty:
        preview = out_of_range[["id", "phq9_pred"]].head(10).to_dict(orient="records")
        raise ValueError(f"ternary.csv phq9_pred must be in [0, 27]. Examples: {preview}")


def validate_official_ids(df: pd.DataFrame, sample_df: pd.DataFrame, file_name: str) -> None:
    """
    Optional check against the official sample file.

    This does not use any hidden labels. It only checks whether submitted IDs
    exactly match the official sample IDs.
    """
    official_ids = set(sample_df["id"].astype(str).tolist())
    submitted_ids = set(df["id"].astype(str).tolist())

    missing_ids = sorted(official_ids - submitted_ids)
    extra_ids = sorted(submitted_ids - official_ids)

    if missing_ids:
        raise ValueError(f"{file_name} is missing official sample IDs: {missing_ids[:10]}")
    if extra_ids:
        raise ValueError(f"{file_name} contains extra IDs not in the official sample file: {extra_ids[:10]}")


def write_submission_zip(binary_df: pd.DataFrame, ternary_df: pd.DataFrame, output_dir: str) -> None:
    """
    Write cleaned binary.csv, ternary.csv, and submission.zip.

    The generated submission.zip will contain binary.csv and ternary.csv
    directly at the root level.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_csv_path = output_dir / "binary.csv"
    ternary_csv_path = output_dir / "ternary.csv"
    zip_path = output_dir / "submission.zip"

    binary_df = binary_df[REQUIRED_BINARY_COLUMNS].copy()
    ternary_df = ternary_df[REQUIRED_TERNARY_COLUMNS].copy()

    binary_df.to_csv(binary_csv_path, index=False, encoding="utf-8")
    ternary_df.to_csv(ternary_csv_path, index=False, encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(binary_csv_path, arcname="binary.csv")
        zf.write(ternary_csv_path, arcname="ternary.csv")

    print(f"Generated: {binary_csv_path}")
    print(f"Generated: {ternary_csv_path}")
    print(f"Generated: {zip_path}")
    print("Done. Please submit submission.zip to CodaBench.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate submission.zip for the MPDD-AVG 2026 CodaBench competition."
    )

    parser.add_argument("--binary_csv", required=True, help="Path to binary.csv")
    parser.add_argument("--ternary_csv", required=True, help="Path to ternary.csv")
    parser.add_argument("--output_dir", default="submission_output", help="Output directory")

    parser.add_argument(
        "--binary_sample",
        default=None,
        help="Optional path to official binary_sample.csv for ID checking",
    )
    parser.add_argument(
        "--ternary_sample",
        default=None,
        help="Optional path to official ternary_sample.csv for ID checking",
    )

    args = parser.parse_args()

    binary_df = read_csv(args.binary_csv)
    ternary_df = read_csv(args.ternary_csv)

    validate_columns(binary_df, REQUIRED_BINARY_COLUMNS, "binary.csv")
    validate_columns(ternary_df, REQUIRED_TERNARY_COLUMNS, "ternary.csv")

    validate_no_duplicate_ids(binary_df, "binary.csv")
    validate_no_duplicate_ids(ternary_df, "ternary.csv")

    validate_id_consistency(binary_df, ternary_df)

    if args.binary_sample:
        binary_sample_df = read_csv(args.binary_sample)
        validate_columns(binary_sample_df, ["id"], "binary_sample.csv")
        validate_official_ids(binary_df, binary_sample_df, "binary.csv")

    if args.ternary_sample:
        ternary_sample_df = read_csv(args.ternary_sample)
        validate_columns(ternary_sample_df, ["id"], "ternary_sample.csv")
        validate_official_ids(ternary_df, ternary_sample_df, "ternary.csv")

    validate_binary_predictions(binary_df)
    validate_ternary_predictions(ternary_df)

    write_submission_zip(binary_df, ternary_df, args.output_dir)


if __name__ == "__main__":
    main()