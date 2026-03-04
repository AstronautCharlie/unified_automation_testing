# This file defines some useful constants and utility functions

import pandas as pd
import os

###############
# Sensor fields
###############

OBJECT_SENSORS = [
    "Accelerometer_CUP_accX",
    "Accelerometer_CUP_accX_2",
    "Accelerometer_CUP_accX_3",
    "Accelerometer_CUP_gyroX",
    "Accelerometer_CUP_gyroY",
    "Accelerometer_SALAMI_accX",
    "Accelerometer_SALAMI_accX_2",
    "Accelerometer_SALAMI_accX_3",
    "Accelerometer_SALAMI_gyroX",
    "Accelerometer_SALAMI_gyroY",
    "Accelerometer_WATER_accX",
    "Accelerometer_WATER_accX_2",
    "Accelerometer_WATER_accX_3",
    "Accelerometer_WATER_gyroX",
    "Accelerometer_WATER_gyroY",
    "Accelerometer_CHEESE_accX",
    "Accelerometer_CHEESE_accX_2",
    "Accelerometer_CHEESE_accX_3",
    "Accelerometer_CHEESE_gyroX",
    "Accelerometer_CHEESE_gyroY",
    "Accelerometer_BREAD_accX",
    "Accelerometer_BREAD_accX_2",
    "Accelerometer_BREAD_accX_3",
    "Accelerometer_BREAD_gyroX",
    "Accelerometer_BREAD_gyroY",
    "Accelerometer_KNIFE1_accX",
    "Accelerometer_KNIFE1_accX_2",
    "Accelerometer_KNIFE1_accX_3",
    "Accelerometer_KNIFE1_gyroX",
    "Accelerometer_KNIFE1_gyroY",
    "Accelerometer_MILK_accX",
    "Accelerometer_MILK_accX_2",
    "Accelerometer_MILK_accX_3",
    "Accelerometer_MILK_gyroX",
    "Accelerometer_MILK_gyroY",
    "Accelerometer_SPOON_accX",
    "Accelerometer_SPOON_accX_2",
    "Accelerometer_SPOON_accX_3",
    "Accelerometer_SPOON_gyroX",
    "Accelerometer_SPOON_gyroY",
    "Accelerometer_SUGAR_accX",
    "Accelerometer_SUGAR_accX_2",
    "Accelerometer_SUGAR_accX_3",
    "Accelerometer_SUGAR_gyroX",
    "Accelerometer_SUGAR_gyroY",
    "Accelerometer_KNIFE2_accX",
    "Accelerometer_KNIFE2_accX_2",
    "Accelerometer_KNIFE2_accX_3",
    "Accelerometer_KNIFE2_gyroX",
    "Accelerometer_KNIFE2_gyroY",
    "Accelerometer_PLATE_accX",
    "Accelerometer_PLATE_accX_2",
    "Accelerometer_PLATE_accX_3",
    "Accelerometer_PLATE_gyroX",
    "Accelerometer_PLATE_gyroY",
    "Accelerometer_GLASS_accX",
    "Accelerometer_GLASS_accX_2",
    "Accelerometer_GLASS_accX_3",
    "Accelerometer_GLASS_gyroX",
    "Accelerometer_GLASS_gyroY",
    "REED_SWITCH_DISHWASHER_S1",
    "REED_SWITCH_FRIDGE_S3",
    "REED_SWITCH_FRIDGE_S2",
    "REED_SWITCH_FRIDGE_S1",
    "REED_SWITCH_MIDDLEDRAWER_S1",
    "REED_SWITCH_MIDDLEDRAWER_S2",
    "REED_SWITCH_MIDDLEDRAWER_S3",
    "REED_SWITCH_LOWERDRAWER_S3",
    "REED_SWITCH_LOWERDRAWER_S2",
    "REED_SWITCH_UPPERDRAWER",
    "REED_SWITCH_DISHWASHER_S3",
    "REED_SWITCH_LOWERDRAWER_S1",
    "REED_SWITCH_DISHWASHER_S2",
    "Accelerometer_DOOR1_accX",
    "Accelerometer_DOOR1_accY",
    "Accelerometer_DOOR1_accZ",
    "Accelerometer_LAZYCHAIR_accX",
    "Accelerometer_LAZYCHAIR_accY",
    "Accelerometer_LAZYCHAIR_accZ",
    "Accelerometer_DOOR2_accX",
    "Accelerometer_DOOR2_accY",
    "Accelerometer_DOOR2_accZ",
    "Accelerometer_DISHWASHER_accX",
    "Accelerometer_DISHWASHER_accY",
    "Accelerometer_DISHWASHER_accZ",
    "Accelerometer_UPPERDRAWER_accX",
    "Accelerometer_UPPERDRAWER_accY",
    "Accelerometer_UPPERDRAWER_accZ",
    "Accelerometer_LOWERDRAWER_accX",
    "Accelerometer_LOWERDRAWER_accY",
    "Accelerometer_LOWERDRAWER_accZ",
    "Accelerometer_MIDDLEDRAWER_accX",
    "Accelerometer_MIDDLEDRAWER_accY",
    "Accelerometer_MIDDLEDRAWER_accZ",
    "Accelerometer_FRIDGE_accX",
    "Accelerometer_FRIDGE_accY",
    "Accelerometer_FRIDGE_accZ"
]

OBJECT_ACCELEROMETERS = [
    "Accelerometer_CUP_accX",
    "Accelerometer_CUP_accX_2",
    "Accelerometer_CUP_accX_3",
    "Accelerometer_CUP_gyroX",
    "Accelerometer_CUP_gyroY",
    "Accelerometer_SALAMI_accX",
    "Accelerometer_SALAMI_accX_2",
    "Accelerometer_SALAMI_accX_3",
    "Accelerometer_SALAMI_gyroX",
    "Accelerometer_SALAMI_gyroY",
    "Accelerometer_WATER_accX",
    "Accelerometer_WATER_accX_2",
    "Accelerometer_WATER_accX_3",
    "Accelerometer_WATER_gyroX",
    "Accelerometer_WATER_gyroY",
    "Accelerometer_CHEESE_accX",
    "Accelerometer_CHEESE_accX_2",
    "Accelerometer_CHEESE_accX_3",
    "Accelerometer_CHEESE_gyroX",
    "Accelerometer_CHEESE_gyroY",
    "Accelerometer_BREAD_accX",
    "Accelerometer_BREAD_accX_2",
    "Accelerometer_BREAD_accX_3",
    "Accelerometer_BREAD_gyroX",
    "Accelerometer_BREAD_gyroY",
    "Accelerometer_KNIFE1_accX",
    "Accelerometer_KNIFE1_accX_2",
    "Accelerometer_KNIFE1_accX_3",
    "Accelerometer_KNIFE1_gyroX",
    "Accelerometer_KNIFE1_gyroY",
    "Accelerometer_MILK_accX",
    "Accelerometer_MILK_accX_2",
    "Accelerometer_MILK_accX_3",
    "Accelerometer_MILK_gyroX",
    "Accelerometer_MILK_gyroY",
    "Accelerometer_SPOON_accX",
    "Accelerometer_SPOON_accX_2",
    "Accelerometer_SPOON_accX_3",
    "Accelerometer_SPOON_gyroX",
    "Accelerometer_SPOON_gyroY",
    "Accelerometer_SUGAR_accX",
    "Accelerometer_SUGAR_accX_2",
    "Accelerometer_SUGAR_accX_3",
    "Accelerometer_SUGAR_gyroX",
    "Accelerometer_SUGAR_gyroY",
    "Accelerometer_KNIFE2_accX",
    "Accelerometer_KNIFE2_accX_2",
    "Accelerometer_KNIFE2_accX_3",
    "Accelerometer_KNIFE2_gyroX",
    "Accelerometer_KNIFE2_gyroY",
    "Accelerometer_PLATE_accX",
    "Accelerometer_PLATE_accX_2",
    "Accelerometer_PLATE_accX_3",
    "Accelerometer_PLATE_gyroX",
    "Accelerometer_PLATE_gyroY",
    "Accelerometer_GLASS_accX",
    "Accelerometer_GLASS_accX_2",
    "Accelerometer_GLASS_accX_3",
    "Accelerometer_GLASS_gyroX",
    "Accelerometer_GLASS_gyroY",
    "Accelerometer_DOOR1_accX",
    "Accelerometer_DOOR1_accY",
    "Accelerometer_DOOR1_accZ",
    "Accelerometer_LAZYCHAIR_accX",
    "Accelerometer_LAZYCHAIR_accY",
    "Accelerometer_LAZYCHAIR_accZ",
    "Accelerometer_DOOR2_accX",
    "Accelerometer_DOOR2_accY",
    "Accelerometer_DOOR2_accZ",
    "Accelerometer_DISHWASHER_accX",
    "Accelerometer_DISHWASHER_accY",
    "Accelerometer_DISHWASHER_accZ",
    "Accelerometer_UPPERDRAWER_accX",
    "Accelerometer_UPPERDRAWER_accY",
    "Accelerometer_UPPERDRAWER_accZ",
    "Accelerometer_LOWERDRAWER_accX",
    "Accelerometer_LOWERDRAWER_accY",
    "Accelerometer_LOWERDRAWER_accZ",
    "Accelerometer_MIDDLEDRAWER_accX",
    "Accelerometer_MIDDLEDRAWER_accY",
    "Accelerometer_MIDDLEDRAWER_accZ",
    "Accelerometer_FRIDGE_accX",
    "Accelerometer_FRIDGE_accY",
    "Accelerometer_FRIDGE_accZ"
]

#################
# Activity Fields
#################
ACTIVITY_FIELDS = [
    'Locomotion',
    'LL_Right_Arm',
    'LL_Right_Arm_Object',
    'LL_Left_Arm',
    'LL_Left_Arm_Object',
    'ML_Both_Arms',
    'HL_Activity'
]

#################
# Activity Values
#################
LL_OBJECTS = [
    'Lazychair', 
    'Fridge', 
    'Drawer1 (top)',
    'Drawer2 (middle)', 
    'Drawer3 (lower)', 
    'Milk', 
    'Sugar', 
    'Table', 
    'Cup', 
    'Spoon', 
    'Knife salami', 
    'Knife cheese', 
    'Bread', 
    'Cheese', 
    'Salami', 
    'Plate'
]

ML_LABELS = [
    'Open Fridge', 
    'Close Fridge', 
    'Open Door 2', 
    'Open Door 1', 
    'Open Drawer 2', 
    'Close Drawer 2', 
    'Open Drawer 1', 
    'Close Drawer 1', 
    'Open Drawer 3', 
    'Close Drawer 3', 
    'Close Door 2', 
    'Toggle Switch', 
    'Close Door 1', 
    'Drink from Cup', 
    'Open Dishwasher', 
    'Close Dishwasher', 
    'Clean Table'
]

HL_LABELS = [
    'Relaxing', 
    'Early morning', 
    'Coffee time', 
    'Sandwich time', 
    'Cleanup'
]

LOCOMOTION_LABELS = [
    'Stand', 
    'Walk',
    'Sit', 
    'Lie'
]

def load_all_adl() -> pd.DataFrame:
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    adl_files = sorted(
        f for f in os.listdir(dataset_dir)
        if "ADL" in f and f.endswith(".parquet")
    )
    dfs = []
    for f in adl_files:
        df = pd.read_parquet(os.path.join(dataset_dir, f))
        stem = f.replace(".parquet", "")       # e.g. "S1-ADL1"
        df["subject"] = stem.split("-")[0]     # e.g. "S1"
        df["recording"] = stem                 # e.g. "S1-ADL1"
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_all_adl_no_label() -> pd.DataFrame: 
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    adl_files = sorted(
        f for f in os.listdir(dataset_dir)
        if "ADL" in f and f.endswith(".parquet")
    )
    dfs = []
    for f in adl_files:
        df = pd.read_parquet(os.path.join(dataset_dir, f))
        stem = f.replace(".parquet", "")       # e.g. "S1-ADL1"
        df["subject"] = stem.split("-")[0]     # e.g. "S1"
        df["recording"] = stem                 # e.g. "S1-ADL1"
        # drop activity columns
        df = df.drop(columns=ACTIVITY_FIELDS)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
