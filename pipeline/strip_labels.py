"""
Strip the 7 label columns (last 7 columns) from all parquet files in ../dataset/.
Each file is overwritten in-place with the label columns removed.

Label columns (columns 244-250 in the original .dat files, 0-indexed 243-249):
  Locomotion, HL_Activity, LL_Left_Arm, LL_Left_Arm_Object,
  LL_Right_Arm, LL_Right_Arm_Object, ML_Both_Arms
"""

from pathlib import Path
import pandas as pd

DATASET_DIR = Path(__file__).parent.parent / "dataset"
NUM_LABEL_COLS = 7


def strip_labels(parquet_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    if df.shape[1] <= NUM_LABEL_COLS:
        print(f"  SKIP {parquet_path.name}: only {df.shape[1]} columns, nothing to strip")
        return

    stripped = df.iloc[:, :-NUM_LABEL_COLS]
    stripped.to_parquet(parquet_path, index=False)
    print(f"  OK   {parquet_path.name}: {df.shape[1]} -> {stripped.shape[1]} columns")


def main() -> None:
    parquet_files = sorted(DATASET_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {DATASET_DIR}")
        return

    print(f"Processing {len(parquet_files)} parquet file(s) in {DATASET_DIR}\n")
    for path in parquet_files:
        strip_labels(path)
    print("\nDone.")


if __name__ == "__main__":
    main()
