"""
Posture estimation from raw IMU sensor data (no locomotion labels).
Estimates time each subject spends in: Standing, Walking, Sitting, Lying.
"""

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
PARQUET_PATH = "../ADL_no_label.parquet"
SAMPLE_RATE_HZ = 30
MS_PER_SAMPLE = 1000 / SAMPLE_RATE_HZ          # ≈33.33 ms

WINDOW_WALK_S = 3                               # rolling window for walking detection
WINDOW_WALK = WINDOW_WALK_S * SAMPLE_RATE_HZ   # samples

WINDOW_SMOOTH_S = 1                             # majority-vote smoothing window
WINDOW_SMOOTH = WINDOW_SMOOTH_S * SAMPLE_RATE_HZ

MIN_LYING_S = 5                                 # minimum duration to accept "lying"
MIN_LYING_SAMPLES = MIN_LYING_S * SAMPLE_RATE_HZ

# Thresholds (milli-g or milli-g·s⁻¹)
THR_LYING_BACK_X = 500      # |BACK_accX| < this → lying
THR_WALKING_ANGVEL = 2000   # rolling mean shoe_angvel > this → walking
THR_SITTING_KNEE_Y = 500    # knee_gravity < this → sitting

# ── Column names ─────────────────────────────────────────────────────────────
BACK_X = "InertialMeasurementUnit_BACK_accX"
BACK_Y = "InertialMeasurementUnit_BACK_accY"
BACK_Z = "InertialMeasurementUnit_BACK_accZ"

L_SHOE_COLS = [
    "InertialMeasurementUnit_L-SHOE_AngVelBodyFrameX",
    "InertialMeasurementUnit_L-SHOE_AngVelBodyFrameY",
    "InertialMeasurementUnit_L-SHOE_AngVelBodyFrameZ",
]
R_SHOE_COLS = [
    "InertialMeasurementUnit_R-SHOE_AngVelBodyFrameX",
    "InertialMeasurementUnit_R-SHOE_AngVelBodyFrameY",
    "InertialMeasurementUnit_R-SHOE_AngVelBodyFrameZ",
]
KNEE_Y = "Accelerometer_RKN^_accY"

POSTURE_LABELS = ["Standing", "Walking", "Sitting", "Lying"]


# ── Step 1: Load & impute ─────────────────────────────────────────────────────
def load_and_impute(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows | subjects: {sorted(df['subject'].unique())} | "
          f"recordings: {sorted(df['recording'].unique())}")

    key_cols = [BACK_X, BACK_Y, BACK_Z] + L_SHOE_COLS + R_SHOE_COLS + [KNEE_Y]
    missing_before = df[key_cols].isna().sum()

    # Forward-fill then backward-fill within each recording to preserve continuity
    df[key_cols] = (
        df.groupby(["subject", "recording"])[key_cols]
        .transform(lambda g: g.ffill().bfill())
    )

    df["imputed_flag"] = missing_before.gt(0).any()  # scalar; per-row not needed
    missing_after = df[key_cols].isna().sum()
    print("Null counts after imputation:")
    print(missing_after[missing_after > 0] if missing_after.any() else "  none")
    return df


# ── Step 2: Feature engineering ───────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # A. Back tilt — not directly used in classifier but useful for inspection
    df["back_tilt"] = np.arctan2(
        np.sqrt(df[BACK_Y] ** 2 + df[BACK_Z] ** 2),
        -df[BACK_X],
    )

    # B. Shoe angular velocity magnitude (instantaneous, per row)
    shoe_sq = pd.concat(
        [df[c] ** 2 for c in L_SHOE_COLS + R_SHOE_COLS], axis=1
    ).mean(axis=1)
    df["shoe_angvel_inst"] = np.sqrt(shoe_sq)

    # Rolling mean of shoe_angvel within each recording (avoids bleed across sessions)
    df["shoe_angvel_roll"] = (
        df.groupby(["subject", "recording"])["shoe_angvel_inst"]
        .transform(lambda g: g.rolling(WINDOW_WALK, min_periods=1, center=True).mean())
    )

    # C. Knee gravity component
    df["knee_gravity"] = df[KNEE_Y]

    return df


# ── Step 3: Classify posture (priority order) ─────────────────────────────────
def classify_posture(df: pd.DataFrame) -> pd.DataFrame:
    posture = pd.Series("Standing", index=df.index, dtype="object")

    lying_mask    = df[BACK_X].abs() < THR_LYING_BACK_X
    walking_mask  = df["shoe_angvel_roll"] > THR_WALKING_ANGVEL
    sitting_mask  = df["knee_gravity"] < THR_SITTING_KNEE_Y

    # Apply in reverse priority so highest-priority overwrites last
    posture[sitting_mask]                       = "Sitting"
    posture[walking_mask]                       = "Walking"
    posture[lying_mask]                         = "Lying"

    df["posture_raw"] = posture
    return df


# ── Step 4: Temporal smoothing ────────────────────────────────────────────────
def majority_vote(series: pd.Series, window: int) -> pd.Series:
    """Rolling majority vote using integer encoding for speed."""
    code, uniques = pd.factorize(series)
    rolled = (
        pd.Series(code, index=series.index)
        .rolling(window, min_periods=1, center=True)
        .apply(lambda x: np.bincount(x.astype(int), minlength=len(uniques)).argmax(),
               raw=True)
    )
    return rolled.map(lambda i: uniques[int(i)])


def enforce_min_lying(series: pd.Series, min_samples: int) -> pd.Series:
    """Convert short 'Lying' runs (< min_samples) to the surrounding label."""
    result = series.copy()
    in_lying = False
    start = 0
    for i, val in enumerate(series):
        if val == "Lying" and not in_lying:
            in_lying = True
            start = i
        elif val != "Lying" and in_lying:
            if (i - start) < min_samples:
                result.iloc[start:i] = series.iloc[start - 1] if start > 0 else "Standing"
            in_lying = False
    # Handle trailing run
    if in_lying and (len(series) - start) < min_samples:
        result.iloc[start:] = series.iloc[start - 1] if start > 0 else "Standing"
    return result


def smooth_posture(df: pd.DataFrame) -> pd.DataFrame:
    smoothed = (
        df.groupby(["subject", "recording"])["posture_raw"]
        .transform(lambda g: majority_vote(g, WINDOW_SMOOTH))
    )
    smoothed = (
        df.groupby(["subject", "recording"])
        .apply(lambda g: enforce_min_lying(smoothed.loc[g.index], MIN_LYING_SAMPLES))
        .reset_index(level=[0, 1], drop=True)
        .sort_index()
    )
    df["posture"] = smoothed
    return df


# ── Step 5: Compute time per posture per subject ──────────────────────────────
def compute_time_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["subject", "posture"])
        .size()
        .rename("n_samples")
        .reset_index()
    )
    counts["seconds"] = counts["n_samples"] * MS_PER_SAMPLE / 1000

    pivot = counts.pivot_table(
        index="subject", columns="posture", values="seconds", fill_value=0
    )
    # Ensure all four columns exist even if a class never appears
    for label in POSTURE_LABELS:
        if label not in pivot.columns:
            pivot[label] = 0.0
    pivot = pivot[POSTURE_LABELS]
    pivot["Total_recorded_s"] = pivot.sum(axis=1)
    return pivot.reset_index()


def compute_time_by_recording(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["subject", "recording", "posture"])
        .size()
        .rename("n_samples")
        .reset_index()
    )
    counts["seconds"] = counts["n_samples"] * MS_PER_SAMPLE / 1000
    pivot = counts.pivot_table(
        index=["subject", "recording"], columns="posture", values="seconds", fill_value=0
    )
    for label in POSTURE_LABELS:
        if label not in pivot.columns:
            pivot[label] = 0.0
    pivot = pivot[POSTURE_LABELS]
    pivot["Total_recorded_s"] = pivot.sum(axis=1)
    return pivot.reset_index()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_and_impute(PARQUET_PATH)
    df = add_features(df)
    df = classify_posture(df)
    df = smooth_posture(df)
    df.to_parquet('pipeline_9b_output_file.parquet')

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = compute_time_table(df)
    print("\n=== Time per posture per subject (seconds) ===")
    print(summary.to_string(index=False, float_format="%.1f"))

    # ── Per-recording breakdown ───────────────────────────────────────────────
    by_rec = compute_time_by_recording(df)
    print("\n=== Per-recording breakdown (seconds) ===")
    print(by_rec.to_string(index=False, float_format="%.1f"))

    # ── Threshold diagnostics ─────────────────────────────────────────────────
    print("\n=== Feature distribution (for threshold validation) ===")
    for col, label in [
        (BACK_X, "|BACK_accX| (lying detector)"),
        ("shoe_angvel_roll", "shoe_angvel_roll (walking detector)"),
        ("knee_gravity", "knee_gravity/RKN^_accY (sitting detector)"),
    ]:
        s = df[col].abs() if col == BACK_X else df[col]
        print(f"\n  {label}")
        print(f"    min={s.min():.1f}  p25={s.quantile(.25):.1f}  "
              f"median={s.median():.1f}  p75={s.quantile(.75):.1f}  max={s.max():.1f}")

    # ── Raw class distribution ────────────────────────────────────────────────
    print("\n=== Overall posture distribution ===")
    dist = (
        df["posture"].value_counts()
        .rename("n_samples")
        .to_frame()
        .assign(seconds=lambda x: x["n_samples"] * MS_PER_SAMPLE / 1000,
                pct=lambda x: 100 * x["n_samples"] / len(df))
    )
    print(dist.to_string(float_format="%.1f"))


if __name__ == "__main__":
    main()
