
from pathlib import Path
from rlearn import RLearn_Model
from rlearn.sports.main_class import PreprocessObservationConfig

config_path = Path("./content/config/preprocessing_fifawc2022.json")

# splitデータのパスを更新
split_data_base_path = "./data/fifawc/preprocess_data/split/"

RLearn_Model(
    state_def="PVS",
    config=config_path,
    num_process=1,
    input_path=split_data_base_path + "train",
    output_path="./data/fifawc/observation_data/split/train",
).run_rlearn(
    run_split_train_test=False,
    run_preprocess_observation=True,
    run_train_and_test=False,
    run_visualize_data=False,
    preprocess_config=PreprocessObservationConfig(batch_size=8)
)

RLearn_Model(
    state_def="PVS",
    config=config_path,
    num_process=1,
    input_path=split_data_base_path + "validation",
    output_path="./data/fifawc/observation_data/split/validation",
).run_rlearn(
    run_split_train_test=False,
    run_preprocess_observation=True,
    run_train_and_test=False,
    run_visualize_data=False,
    preprocess_config=PreprocessObservationConfig(batch_size=8)
)

RLearn_Model(
    state_def="PVS",
    config=config_path,
    num_process=1,
    input_path=split_data_base_path + "test",
    output_path="./data/fifawc/observation_data/split/test",
).run_rlearn(
    run_split_train_test=False,
    run_preprocess_observation=True,
    run_train_and_test=False,
    run_visualize_data=False,
    preprocess_config=PreprocessObservationConfig(batch_size=8)
)

print("\n✅ Creating Observation data completed successfully!")