import re
import pandas as pd
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "secrets"


def parse_column_names(path: Path) -> list[str]:
    names = {}
    for line in path.read_text().splitlines():
        m = re.match(r"Column:\s+(\d+)\s+(.+)", line)
        if m:
            col_idx = int(m.group(1))
            # Extract name between "Column: N " and the first semicolon (or end of line)
            raw_name = m.group(2).split(";")[0].strip().replace(" ", "_")
            names[col_idx] = raw_name
    raw = [names[i] for i in sorted(names)]
    # Deduplicate: append _2, _3, ... for repeated names
    seen: dict[str, int] = {}
    result = []
    for name in raw:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            result.append(name)
    return result


def parse_label_legend(path: Path) -> dict[str, dict[int, str]]:
    mappings: dict[str, dict[int, str]] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"(\d+)\s+-\s+(\S+)\s+-\s+(.+)", line)
        if m:
            code = int(m.group(1))
            track = m.group(2)
            label = m.group(3).strip()
            mappings.setdefault(track, {})[code] = label
    return mappings


LABEL_COLS = [
    "Locomotion",
    "HL_Activity",
    "LL_Left_Arm",
    "LL_Left_Arm_Object",
    "LL_Right_Arm",
    "LL_Right_Arm_Object",
    "ML_Both_Arms",
]


def main():
    columns = parse_column_names(DATASET_DIR / "column_names.txt")
    label_maps = parse_label_legend(DATASET_DIR / "label_legend.txt")

    dat_files = sorted(DATASET_DIR.glob("*.dat"))
    print(f"Found {len(dat_files)} .dat files")

    for dat_path in dat_files:
        print(f"Processing {dat_path.name} ...", end=" ", flush=True)
        df = pd.read_csv(dat_path, sep=" ", header=None, names=columns)

        for col in LABEL_COLS:
            if col in df.columns and col in label_maps:
                df[col] = (
                    df[col]
                    .astype("Int64")  # nullable int to handle NaN
                    .map(label_maps[col])  # map int codes → strings (unmapped → NaN)
                )

        out_path = dat_path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        print(f"saved -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
