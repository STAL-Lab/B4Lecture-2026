
from pathlib import Path
import os
from rlearn import RLearn_Model
from rlearn.sports.main_class import VisualizeDataConfig

config_path = Path("./content/config/exp_fifawc2022.json")

RLearn_Model(
    state_def="PVS"
).run_rlearn(
    run_split_train_test=False,
    run_preprocess_observation=False,
    run_train_and_test=False,
    run_visualize_data=True,
    visualize_config=VisualizeDataConfig(
        exp_config_path=config_path,
        checkpoint_path="./content/output_no_class_weight/sarsa_attacker/no_class_weight/checkpoints/best-04-30-val_loss=0.0771.ckpt",
        model_name="exp_fifawc2022_no_class_weight",
        tracking_file_path="./data/fifawc/preprocess_data/3814/events.jsonl",
        match_id="3814",
        sequence_id=0,
        viz_style="bar",
        movie_output_dir="./content/output_no_class_weight",
        keep_frames=True
    ),
)

print("\n✅ Updated visualize_qvalues.py without unsupported argument!")