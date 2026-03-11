"""
Identify which sensor readings are most predictive of each class
in the activity columns of the OPPORTUNITY UCI dataset.

Approach:
- Load all ADL parquet files (not Drill)
- For each activity column:
    - Impute missing sensor values with column medians (computed on labeled rows)
    - Train a multiclass Random Forest to get overall feature importances
    - For each class, train a binary one-vs-rest Random Forest to get class-specific importances
- Report top sensors per class
"""

import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

ACTIVITY_COLS = [
    "Locomotion",
    "HL_Activity",
    "LL_Left_Arm",
    "LL_Left_Arm_Object",
    "LL_Right_Arm",
    "LL_Right_Arm_Object",
    "ML_Both_Arms",
]

TOP_N = 10  # Top sensors to report per class
MIN_CLASS_SAMPLES = 100  # Skip classes with fewer samples than this

def load_adl_data(dataset_dir="dataset"):
    files = sorted(glob.glob(f"{dataset_dir}/S*-ADL*.parquet"))
    print(f"Loading {len(files)} ADL files...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df):,}, columns: {len(df.columns)}")
    return df


def get_sensor_cols(df):
    non_sensor = {"MILLISEC"} | set(ACTIVITY_COLS)
    return [c for c in df.columns if c not in non_sensor]


def impute_sensors(X: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    """Fill NaN in sensor columns with precomputed medians."""
    X_filled = X.copy()
    for col in X.columns:
        if X_filled[col].isna().any():
            X_filled[col] = X_filled[col].fillna(medians[col])
    return X_filled.values.astype(np.float32)


def train_rf(X, y, class_weight="balanced", n_estimators=100, random_state=42):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
        max_features="sqrt",
    )
    clf.fit(X, y)
    return clf


def analyze_activity(df, sensor_cols, activity_col):
    print(f"\n{'='*60}")
    print(f"Activity: {activity_col}")
    print(f"{'='*60}")

    # Only use rows where this activity is labeled
    labeled = df[df[activity_col].notna()].copy()
    print(f"  Labeled rows: {len(labeled):,} / {len(df):,}")

    classes = sorted(labeled[activity_col].unique())
    print(f"  Classes ({len(classes)}): {classes}")

    X_df = labeled[sensor_cols]
    medians = X_df.median()  # computed only on labeled rows

    X = impute_sensors(X_df, medians)
    y_raw = labeled[activity_col].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    results = {}

    # --- Multiclass RF: overall feature importance ---
    print(f"  Training multiclass RF...")
    clf_multi = train_rf(X, y)
    importances = clf_multi.feature_importances_
    print(importances)
    ranked_idx = np.argsort(importances)[::-1]
    print(ranked_idx)

    results["_overall"] = {
        "top_sensors": [
            (sensor_cols[i], float(importances[i]))
            for i in ranked_idx[:TOP_N]
        ]
    }

    # --- One-vs-rest RF per class ---
    for cls_encoded, cls_name in enumerate(le.classes_):
        cls_count = int((y == cls_encoded).sum())
        if cls_count < MIN_CLASS_SAMPLES:
            print(f"  Skipping class '{cls_name}' (only {cls_count} samples)")
            continue

        y_binary = (y == cls_encoded).astype(int)
        print(f"  Training OvR RF for class '{cls_name}' (n={cls_count:,})...")
        clf_ovr = train_rf(X, y_binary, class_weight="balanced")
        imp = clf_ovr.feature_importances_
        ranked = np.argsort(imp)[::-1]
        results[cls_name] = {
            "n_samples": cls_count,
            "top_sensors": [
                (sensor_cols[i], float(imp[i]))
                for i in ranked[:TOP_N]
            ]
        }

    return results


def print_results(all_results):
    for activity, class_results in all_results.items():
        print(f"\n{'#'*70}")
        print(f"# {activity}")
        print(f"{'#'*70}")

        overall = class_results.get("_overall", {})
        if overall:
            print(f"\n  [Overall most discriminative sensors]")
            for rank, (sensor, score) in enumerate(overall["top_sensors"], 1):
                print(f"    {rank:2d}. {sensor:<55} {score:.4f}")

        for cls_name, info in class_results.items():
            if cls_name == "_overall":
                continue
            print(f"\n  [Class: '{cls_name}'] (n={info['n_samples']:,})")
            for rank, (sensor, score) in enumerate(info["top_sensors"], 1):
                print(f"    {rank:2d}. {sensor:<55} {score:.4f}")


def save_results_csv(all_results, out_path="sensor_importance_by_class.csv"):
    rows = []
    for activity, class_results in all_results.items():
        for cls_name, info in class_results.items():
            for rank, (sensor, score) in enumerate(info["top_sensors"], 1):
                rows.append({
                    "activity_column": activity,
                    "class": cls_name,
                    "rank": rank,
                    "sensor": sensor,
                    "importance": score,
                    "n_samples": info.get("n_samples", None),
                })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    return out_df


def main():
    df = load_adl_data()
    sensor_cols = get_sensor_cols(df)
    print(f"Sensor columns: {len(sensor_cols)}")

    all_results = {}
    for activity_col in ACTIVITY_COLS:
        all_results[activity_col] = analyze_activity(df, sensor_cols, activity_col)

    print_results(all_results)
    save_results_csv(all_results)


if __name__ == "__main__":
    main()
