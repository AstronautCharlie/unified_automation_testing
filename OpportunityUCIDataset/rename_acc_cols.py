import pandas as pd
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "secrets"


def rename_col(col: str) -> str:
    if col.endswith("accX_2"):
        return col[:-len("accX_2")] + "accY"
    if col.endswith("accX_3"):
        return col[:-len("accX_3")] + "accZ"
    return col


for pq in sorted(DATASET_DIR.glob("*.parquet")):
    df = pd.read_parquet(pq)
    renamed = {col: rename_col(col) for col in df.columns if rename_col(col) != col}
    df.rename(columns=renamed, inplace=True)
    df.to_parquet(pq, index=False)
    print(f"{pq.name}: renamed {len(renamed)} cols")

print("Done.")
