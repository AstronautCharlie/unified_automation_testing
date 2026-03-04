"""
Posture/Activity Classification: Stand, Sit, Walk, Lie
=======================================================

This script classifies each timestep in ADL_no_label.parquet as one of:
    'Walk', 'Stand', 'Sit', 'Lie'

Using only body-worn sensor data (columns 1-133).

Physical approach:
------------------
Three key physical signals drive classification:

1. WALKING DETECTION — foot angular velocity
   The shoe IMUs (L-SHOE, R-SHOE) measure angular velocity in the body
   frame. During walking, the foot swings forward and back through large
   arcs every ~0.5–1.0 s, generating sustained high angular velocity.
   During standing/sitting/lying, feet are stationary → low angular
   velocity. A 3-second rolling mean of the peak shoe angular velocity
   separates sustained walking from brief foot adjustments or fidgeting.

2. LYING DETECTION — back inclination from vertical
   The IMU on the back measures acceleration. At rest, this is dominated
   by gravity. The BACK sensor's X-axis is roughly aligned with gravity
   when the person is upright (empirical mean ≈ −862 milli-g ≈ −0.86g
   when standing). We compute the angle of this axis from vertical.
   When lying down, the torso rotates ~90°, pushing the tilt above 60°.
   We additionally require low foot motion (shoe_angvel < 500) to avoid
   confusing a forward bend during walking or reaching with lying down.

3. SIT vs. STAND — knee/thigh orientation
   An accelerometer near the right knee (RKN^) measures gravity along
   the length of the thigh. When standing, the thigh is vertical and the
   sensor's Y-axis carries most of the gravitational signal (~900–1000
   milli-g). When sitting, the thigh rotates to ~90° horizontal, so the
   Y-axis gravity component drops to ~200–500 milli-g. A threshold at
   650 milli-g separates the two postures. This bimodal split is clearly
   visible in the data: roughly 40% of stationary-upright samples fall
   below 650 (sitting), 60% above (standing).

Classification pipeline:
------------------------
Step 1: Compute instantaneous features (per row)
Step 2: Per-recording rolling smoothing (3-second window)
Step 3: Apply rules in priority order: Walk → Lie → Sit → Stand
Step 4: Temporal majority-vote smoothing (3-second window)
Step 5: Enforce minimum Lie duration (5 s) to remove brief false positives
        caused by forward bends or transitional movements
"""

import pandas as pd
import numpy as np
from scipy import stats


# ── Configuration ────────────────────────────────────────────────────────────

ROLL_WINDOW_SAMPLES = 90    # ~3 s at 30 Hz — used for all rolling features

WALK_ANGVEL_THRESH = 2000   # milli-deg/s: rolling-mean shoe angvel > this → Walk

LIE_BACK_TILT_THRESH = 60   # degrees: back tilt from vertical > this → candidate Lie
LIE_SHOE_ANGVEL_MAX = 500   # shoe angvel must be below this to confirm Lie (not bending)

SIT_KNEE_Y_THRESH = 650     # milli-g: knee Y < this → Sit (thigh is horizontal)

SMOOTH_WINDOW_SAMPLES = 90  # ~3 s majority-vote window for temporal smoothing

LIE_MIN_SAMPLES = 150       # ~5 s: Lie segments shorter than this → revert to Stand


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_parquet("dataset/ADL_no_label.parquet")
print(f"  Shape: {df.shape}")
print(f"  Recordings: {sorted(df['recording'].unique())}")


# ── Feature computation ───────────────────────────────────────────────────────

print("\nComputing features...")

# Feature 1: Shoe angular velocity magnitude — max of left/right shoe
shoe_angvel_L = np.sqrt(
    df["InertialMeasurementUnit_L-SHOE_AngVelBodyFrameX"] ** 2
    + df["InertialMeasurementUnit_L-SHOE_AngVelBodyFrameY"] ** 2
    + df["InertialMeasurementUnit_L-SHOE_AngVelBodyFrameZ"] ** 2
)
shoe_angvel_R = np.sqrt(
    df["InertialMeasurementUnit_R-SHOE_AngVelBodyFrameX"] ** 2
    + df["InertialMeasurementUnit_R-SHOE_AngVelBodyFrameY"] ** 2
    + df["InertialMeasurementUnit_R-SHOE_AngVelBodyFrameZ"] ** 2
)
df["shoe_angvel"] = np.maximum(shoe_angvel_L.fillna(0), shoe_angvel_R.fillna(0))

# Feature 2: Back inclination from vertical
# BACK IMU accX: dominant gravity axis when upright (mean ≈ −862 milli-g standing)
# tilt = arccos(−accX / |acc|); upright → 0°, horizontal → 90°
back_x = df["InertialMeasurementUnit_BACK_accX"]
back_y = df["InertialMeasurementUnit_BACK_accY"]
back_z = df["InertialMeasurementUnit_BACK_accZ"]
back_mag = np.sqrt(back_x ** 2 + back_y ** 2 + back_z ** 2)
df["back_tilt"] = np.degrees(
    np.arccos(np.clip(-back_x / back_mag.where(back_mag > 10, np.nan), -1, 1))
)

# Feature 3: Knee thigh-axis gravity (RKN^ sensor Y-axis)
# High when thigh is vertical (standing), low when thigh is horizontal (sitting)
df["knee_y"] = df["Accelerometer_RKN^_accY"]


# ── Per-recording rolling smoothing ──────────────────────────────────────────

print("Applying rolling smoothing per recording...")
recordings = sorted(df["recording"].unique())
df["shoe_angvel_roll"] = np.nan
df["back_tilt_roll"] = np.nan
df["knee_y_roll"] = np.nan

for rec in recordings:
    mask = df["recording"] == rec
    rec_df = df.loc[mask].sort_values("MILLISEC")
    idx = rec_df.index

    shoe = rec_df["shoe_angvel"].ffill().bfill()
    bt = rec_df["back_tilt"].ffill().bfill()
    ky = rec_df["knee_y"].ffill().bfill()

    # Rolling mean for shoe angvel: captures sustained locomotion patterns
    df.loc[idx, "shoe_angvel_roll"] = (
        shoe.rolling(ROLL_WINDOW_SAMPLES, center=True, min_periods=5).mean().values
    )
    # Rolling median for posture: robust to transient spikes from quick movements
    df.loc[idx, "back_tilt_roll"] = (
        bt.rolling(ROLL_WINDOW_SAMPLES, center=True, min_periods=5).median().values
    )
    df.loc[idx, "knee_y_roll"] = (
        ky.rolling(ROLL_WINDOW_SAMPLES, center=True, min_periods=5).median().values
    )


# ── Classification rules ──────────────────────────────────────────────────────

print("Classifying postures...")

def classify(shoe_angvel, back_tilt, knee_y):
    """
    Priority order: Walk > Lie > Sit > Stand

    Walk:  sustained foot motion (high angular velocity) while back is upright.
    Lie:   back near-horizontal AND feet stationary (rules out forward bends).
    Sit:   stationary, back upright, but thigh horizontal (low knee Y gravity).
    Stand: default — stationary, back upright, thigh vertical.
    """
    walk = (shoe_angvel > WALK_ANGVEL_THRESH) & (back_tilt < LIE_BACK_TILT_THRESH)
    lie  = (back_tilt >= LIE_BACK_TILT_THRESH) & (shoe_angvel < LIE_SHOE_ANGVEL_MAX)
    sit  = (~walk) & (~lie) & (knee_y < SIT_KNEE_Y_THRESH)

    label = pd.Series("Stand", index=shoe_angvel.index)
    label[sit]  = "Sit"
    label[lie]  = "Lie"
    label[walk] = "Walk"   # Walk overrides Lie/Sit/Stand
    return label


shoe_f = df["shoe_angvel_roll"].fillna(df["shoe_angvel"])
back_f = df["back_tilt_roll"].fillna(df["back_tilt"])
knee_f = df["knee_y_roll"].fillna(df["knee_y"]).fillna(1000)  # default: knee upright

df["label_raw"] = classify(shoe_f, back_f, knee_f)


# ── Temporal majority-vote smoothing ─────────────────────────────────────────

print("Applying temporal majority-vote smoothing...")

LABEL_MAP = {"Walk": 0, "Lie": 1, "Sit": 2, "Stand": 3}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}

df["label_int"] = df["label_raw"].map(LABEL_MAP)
df["label_smooth"] = np.nan

for rec in recordings:
    mask = df["recording"] == rec
    rec_df = df.loc[mask].sort_values("MILLISEC")
    idx = rec_df.index

    smoothed = (
        rec_df["label_int"]
        .rolling(SMOOTH_WINDOW_SAMPLES, center=True, min_periods=5)
        .apply(lambda x: stats.mode(x, keepdims=True)[0][0], raw=True)
    )
    df.loc[idx, "label_smooth"] = smoothed.values

df["label_smooth"] = (
    df["label_smooth"].fillna(df["label_int"]).astype(int).map(LABEL_INV)
)


# ── Enforce minimum Lie duration ──────────────────────────────────────────────

print("Enforcing minimum Lie duration (remove brief false positives)...")

df["label"] = df["label_smooth"].copy()

for rec in recordings:
    mask = df["recording"] == rec
    rec_idx = df.index[mask]
    labels = df.loc[rec_idx, "label"].values

    i = 0
    while i < len(labels):
        if labels[i] == "Lie":
            j = i
            while j < len(labels) and labels[j] == "Lie":
                j += 1
            if (j - i) < LIE_MIN_SAMPLES:
                labels[i:j] = "Stand"  # brief Lie → revert to Stand
            i = j
        else:
            i += 1

    df.loc[rec_idx, "label"] = labels


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n=== Classification Results ===")
counts = df["label"].value_counts()
total = len(df)
for lbl in ["Walk", "Stand", "Sit", "Lie"]:
    n = counts.get(lbl, 0)
    print(f"  {lbl:5s}: {n:7,} rows ({n / total * 100:5.1f}%)")

print("\nPer-recording breakdown (% of time):")
pivot = df.groupby(["recording", "label"]).size().unstack(fill_value=0)
cols = [c for c in ["Walk", "Stand", "Sit", "Lie"] if c in pivot.columns]
pivot_pct = (pivot[cols].T / pivot[cols].sum(axis=1)).T * 100
print(pivot_pct.round(1).to_string())

print("\nTransition counts per recording (fewer = smoother labels):")
for rec in recordings:
    sub = df[df["recording"] == rec].sort_values("MILLISEC")
    n_trans = (sub["label"] != sub["label"].shift()).sum() - 1
    print(f"  {rec}: {n_trans}")


# ── Save results ──────────────────────────────────────────────────────────────

out_cols = ["MILLISEC", "subject", "recording", "label"]
df[out_cols].to_parquet("prompt_6/labels.parquet", index=False)
print(f"\nSaved labels to prompt_6/labels.parquet")

df[out_cols].sample(10000, random_state=42).sort_values(
    ["recording", "MILLISEC"]
).to_csv("prompt_6/labels_sample.csv", index=False)
print("Saved sample to prompt_6/labels_sample.csv")
