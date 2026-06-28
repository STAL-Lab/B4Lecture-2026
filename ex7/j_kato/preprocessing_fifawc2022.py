
from pathlib import Path
from preprocessing import SAR_data

# Step 2: Define your data path as a Path object
data_directory = Path("./data/fifawc")
config_path = Path("./content/config/preprocessing_fifawc2022.json")
# Step 3: Pass the Path object to the class
SAR_data(
    data_provider="fifawc",
    data_path=data_directory,
    state_def="PVS",
    config_path=config_path,
    preprocess_method="SAR",
    match_id=3812,
    max_workers=1
).preprocess_data()

SAR_data(
    data_provider="fifawc",
    data_path=data_directory,
    state_def="PVS",
    config_path=config_path,
    preprocess_method="SAR",
    match_id=3813,
    max_workers=1
).preprocess_data()

SAR_data(
    data_provider="fifawc",
    data_path=data_directory,
    state_def="PVS",
    config_path=config_path,
    preprocess_method="SAR",
    match_id=3814,
    max_workers=1
).preprocess_data()

print("\n✅ Preprocessing completed successfully!")