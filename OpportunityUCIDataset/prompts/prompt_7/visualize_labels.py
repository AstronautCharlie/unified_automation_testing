"""
Visualization of posture/activity labels.

Produces:
  prompt_7/label_timeseries.png — stacked time series for two recordings
  prompt_7/label_distribution.png — bar chart of label proportions
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load ──────────────────────────────────────────────────────────────────────

df_raw = pd.read_parquet("dataset/ADL_no_label.parquet")
labels = pd.read_parquet("prompt_7/labels.parquet")
df = df_raw.merge(labels[["MILLISEC", "recording", "label"]], on=["MILLISEC", "recording"])

COLORS = {"Walk": "#2196F3", "Stand": "#4CAF50", "Sit": "#FF9800", "Lie": "#9C27B0"}
LABEL_ORDER = ["Walk", "Stand", "Sit", "Lie"]


# ── Figure 1: Time series for two contrasting recordings ──────────────────────

fig, axes = plt.subplots(4, 2, figsize=(18, 12), sharex=False)
fig.suptitle("Activity Labels + Key Sensor Signals", fontsize=14, fontweight="bold")

RECS = ["S1-ADL1", "S3-ADL1"]

for col, rec in enumerate(RECS):
    sub = df[df["recording"] == rec].sort_values("MILLISEC").copy()
    t = (sub["MILLISEC"] - sub["MILLISEC"].iloc[0]) / 1000  # seconds

    # Panel 0: label as colour band
    ax = axes[0, col]
    label_num = sub["label"].map({"Walk": 0, "Stand": 1, "Sit": 2, "Lie": 3})
    for i in range(len(sub) - 1):
        lbl = sub["label"].iloc[i]
        ax.axvspan(t.iloc[i], t.iloc[i + 1], color=COLORS[lbl], alpha=0.85, linewidth=0)
    ax.set_yticks([])
    ax.set_title(f"{rec} — Activity Label", fontsize=11)
    ax.set_xlim(t.iloc[0], t.iloc[-1])

    # Panel 1: shoe angular velocity
    ax = axes[1, col]
    shoe_L = np.sqrt(
        sub["InertialMeasurementUnit_L-SHOE_AngVelBodyFrameX"] ** 2
        + sub["InertialMeasurementUnit_L-SHOE_AngVelBodyFrameY"] ** 2
        + sub["InertialMeasurementUnit_L-SHOE_AngVelBodyFrameZ"] ** 2
    )
    shoe_R = np.sqrt(
        sub["InertialMeasurementUnit_R-SHOE_AngVelBodyFrameX"] ** 2
        + sub["InertialMeasurementUnit_R-SHOE_AngVelBodyFrameY"] ** 2
        + sub["InertialMeasurementUnit_R-SHOE_AngVelBodyFrameZ"] ** 2
    )
    ax.plot(t, np.maximum(shoe_L, shoe_R), color="#2196F3", lw=0.4, alpha=0.8)
    ax.axhline(2000, color="red", lw=1, ls="--", label="Walk threshold")
    ax.set_ylabel("Shoe AngVel\n(milli-deg/s)", fontsize=9)
    ax.set_ylim(0, 8000)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.legend(fontsize=8)

    # Panel 2: back tilt from vertical
    ax = axes[2, col]
    bx = sub["InertialMeasurementUnit_BACK_accX"]
    by = sub["InertialMeasurementUnit_BACK_accY"]
    bz = sub["InertialMeasurementUnit_BACK_accZ"]
    bmag = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    bt = np.degrees(np.arccos(np.clip(-bx / bmag.where(bmag > 10, np.nan), -1, 1)))
    ax.plot(t, bt, color="#E91E63", lw=0.4, alpha=0.8)
    ax.axhline(60, color="purple", lw=1, ls="--", label="Lie threshold")
    ax.set_ylabel("Back Tilt\n(degrees from vertical)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.legend(fontsize=8)

    # Panel 3: knee Y gravity
    ax = axes[3, col]
    ax.plot(t, sub["Accelerometer_RKN^_accY"], color="#FF9800", lw=0.4, alpha=0.8)
    ax.axhline(650, color="brown", lw=1, ls="--", label="Sit threshold")
    ax.set_ylabel("Knee Y\n(milli-g)", fontsize=9)
    ax.set_ylim(-500, 1500)
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.legend(fontsize=8)

# Legend for label colours
patches = [mpatches.Patch(color=COLORS[l], label=l) for l in LABEL_ORDER]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=11,
           title="Activity label", title_fontsize=10, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("prompt_7/label_timeseries.png", dpi=150, bbox_inches="tight")
print("Saved prompt_7/label_timeseries.png")


# ── Figure 2: Distribution by recording ──────────────────────────────────────

pivot = labels.groupby(["recording", "label"]).size().unstack(fill_value=0)
cols = [c for c in LABEL_ORDER if c in pivot.columns]
pivot_pct = (pivot[cols].T / pivot[cols].sum(axis=1)).T * 100

fig2, ax2 = plt.subplots(figsize=(14, 6))
bottom = np.zeros(len(pivot_pct))
for lbl in LABEL_ORDER:
    if lbl in pivot_pct.columns:
        vals = pivot_pct[lbl].values
        ax2.bar(pivot_pct.index, vals, bottom=bottom, color=COLORS[lbl], label=lbl)
        bottom += vals

ax2.set_xlabel("Recording", fontsize=11)
ax2.set_ylabel("% of time", fontsize=11)
ax2.set_title("Activity Label Distribution per Recording", fontsize=13)
ax2.set_xticks(range(len(pivot_pct)))
ax2.set_xticklabels(pivot_pct.index, rotation=45, ha="right", fontsize=9)
ax2.legend(title="Activity", loc="upper right", fontsize=10)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("prompt_7/label_distribution.png", dpi=150, bbox_inches="tight")
print("Saved prompt_7/label_distribution.png")
