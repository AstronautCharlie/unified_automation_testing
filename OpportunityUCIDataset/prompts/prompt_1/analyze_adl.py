"""
ADL Parquet Analysis: Sensor-Activity Relationship Study
Opportunity UCI Dataset
"""

import os
import glob
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 1. Load & concatenate all 20 ADL parquet files
# ─────────────────────────────────────────────
DATASET_DIR = "/Users/trevor/Documents/code/data/OpportunityUCIDataset/dataset"
files = sorted(glob.glob(os.path.join(DATASET_DIR, "*-ADL*.parquet")))
print(f"Found {len(files)} ADL parquet files.")

dfs = []
for f in files:
    df_tmp = pd.read_parquet(f)
    df_tmp["_source_file"] = os.path.basename(f)
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]} ... ({len(df.columns)} total)")

# ─────────────────────────────────────────────
# 2. Identify label columns and sensor columns
# ─────────────────────────────────────────────
LABEL_COLS = [
    "Locomotion",
    "HL_Activity",
    "LL_Left_Arm",
    "LL_Left_Arm_Object",
    "LL_Right_Arm",
    "LL_Right_Arm_Object",
    "ML_Both_Arms",
]

# Sensor columns = all numeric columns that are NOT label cols and not metadata
non_sensor = set(LABEL_COLS) | {"_source_file"}
sensor_cols = [c for c in df.columns if c not in non_sensor and pd.api.types.is_numeric_dtype(df[c])]
print(f"\nLabel columns: {LABEL_COLS}")
print(f"Sensor columns: {len(sensor_cols)} total")
print(f"Sample sensor cols: {sensor_cols[:15]}")

# ─────────────────────────────────────────────
# 3. Class balance per label
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("CLASS BALANCE")
print("="*70)

class_balance = {}
for label in LABEL_COLS:
    if label not in df.columns:
        print(f"  WARNING: {label} not found in dataframe")
        continue
    counts = df[label].value_counts(dropna=False).sort_index()
    class_balance[label] = counts
    print(f"\n{label}:")
    for cls, cnt in counts.items():
        pct = 100 * cnt / len(df)
        print(f"    {str(cls):30s}  {cnt:7d}  ({pct:5.1f}%)")

# ─────────────────────────────────────────────
# 4. F-statistic computation (numpy/pandas only)
#    F ≈ variance_of_group_means / mean_of_group_variances
# ─────────────────────────────────────────────

def compute_f_stats(df, sensor_cols, label_col, min_group_size=30):
    """
    For each sensor column, compute an approximate one-way F-statistic
    across the groups defined by label_col.

    F ≈ var(group_means) / mean(group_vars)

    Returns a Series of F-scores indexed by sensor name, sorted descending.
    """
    col = df[label_col].dropna()
    valid_idx = col.index
    groups = col.unique()
    # Filter groups with enough samples
    group_masks = {}
    for g in groups:
        mask = (df[label_col] == g)
        if mask.sum() >= min_group_size:
            group_masks[g] = mask

    if len(group_masks) < 2:
        return pd.Series(dtype=float)

    f_scores = {}
    for sensor in sensor_cols:
        s = df[sensor]
        group_means = []
        group_vars = []
        for g, mask in group_masks.items():
            vals = s[mask].dropna()
            if len(vals) < 2:
                continue
            group_means.append(vals.mean())
            group_vars.append(vals.var())

        if len(group_means) < 2:
            f_scores[sensor] = np.nan
            continue

        gm = np.array(group_means)
        gv = np.array(group_vars)
        numerator = np.var(gm, ddof=1)        # variance of group means
        denominator = np.mean(gv)             # mean of within-group variances
        if denominator == 0 or np.isnan(denominator):
            f_scores[sensor] = np.nan
        else:
            f_scores[sensor] = numerator / denominator

    return pd.Series(f_scores).dropna().sort_values(ascending=False)


print("\n" + "="*70)
print("TOP DISCRIMINATIVE SENSORS PER LABEL (F-statistic)")
print("="*70)

TOP_N = 10
top_sensors_per_label = {}

for label in LABEL_COLS:
    if label not in df.columns:
        continue
    print(f"\nComputing F-stats for: {label} ...")
    f_scores = compute_f_stats(df, sensor_cols, label)
    top = f_scores.head(TOP_N)
    top_sensors_per_label[label] = top
    print(f"  Top {TOP_N} sensors:")
    for sensor, score in top.items():
        print(f"    {sensor:55s}  F={score:10.2f}")

# ─────────────────────────────────────────────
# 5. Per-class mean of top discriminative sensors
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("PER-CLASS MEAN OF TOP DISCRIMINATIVE SENSORS")
print("="*70)

per_class_means = {}

for label in LABEL_COLS:
    if label not in df.columns or label not in top_sensors_per_label:
        continue
    top_sensor_list = list(top_sensors_per_label[label].index)
    print(f"\n{'─'*60}")
    print(f"Label: {label}")
    print(f"{'─'*60}")

    subset = df[[label] + top_sensor_list].dropna(subset=[label])
    group_means = subset.groupby(label)[top_sensor_list].mean()
    per_class_means[label] = group_means

    # Print rounded for readability
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: f'{x:10.2f}')
    print(group_means.to_string())

# ─────────────────────────────────────────────
# 6. Cross-label sensor overlap analysis
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("CROSS-LABEL SENSOR OVERLAP")
print("="*70)

# Build a dict of sets
label_sensor_sets = {}
for label, top in top_sensors_per_label.items():
    label_sensor_sets[label] = set(top.index)

# For each pair of labels, compute intersection
labels_present = list(label_sensor_sets.keys())
for i in range(len(labels_present)):
    for j in range(i+1, len(labels_present)):
        l1, l2 = labels_present[i], labels_present[j]
        overlap = label_sensor_sets[l1] & label_sensor_sets[l2]
        if overlap:
            print(f"\n{l1}  <->  {l2}:")
            for s in sorted(overlap):
                print(f"    {s}")

# ─────────────────────────────────────────────
# 7. Sensor group classification helper
# ─────────────────────────────────────────────

def classify_sensor(name):
    """Return a high-level sensor group for a column name."""
    n = name.lower()
    if 'acc' in n:
        return 'Accelerometer'
    elif 'gyro' in n or 'rz' in n or 'ry' in n or 'rx' in n:
        return 'Gyroscope/IMU'
    elif 'mag' in n or 'compass' in n:
        return 'Magnetometer'
    elif 'reed' in n:
        return 'Reed switch'
    elif 'tag' in n or 'location' in n or 'ir' in n:
        return 'Location tag'
    elif 'object' in n or 'obj' in n:
        return 'Object sensor'
    elif 'quaternion' in n or 'quat' in n:
        return 'Quaternion/IMU'
    else:
        return 'Other/Unknown'

print("\n" + "="*70)
print("SENSOR GROUP BREAKDOWN PER LABEL")
print("="*70)

for label, top in top_sensors_per_label.items():
    print(f"\n{label}:")
    from collections import Counter
    group_counter = Counter(classify_sensor(s) for s in top.index)
    for grp, cnt in group_counter.most_common():
        sensors_in_grp = [s for s in top.index if classify_sensor(s) == grp]
        print(f"  {grp}: {cnt}")
        for s in sensors_in_grp:
            print(f"    - {s}")

# ─────────────────────────────────────────────
# 8. Global top sensors across all labels
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("GLOBAL SENSOR IMPORTANCE (sum of F-scores across labels)")
print("="*70)

all_f = {}
for label, top in top_sensors_per_label.items():
    for sensor, score in top.items():
        all_f[sensor] = all_f.get(sensor, 0.0) + score

global_ranking = pd.Series(all_f).sort_values(ascending=False)
print(f"\nTop 20 sensors by cumulative F-score across all labels:")
for sensor, score in global_ranking.head(20).items():
    grp = classify_sensor(sensor)
    count = sum(1 for lbl in top_sensors_per_label if sensor in top_sensors_per_label[lbl].index)
    print(f"  {sensor:55s}  cumF={score:10.2f}  group={grp}  appears_in={count} labels")

print("\n\nANALYSIS COMPLETE.")
