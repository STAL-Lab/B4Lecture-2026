import json
from pathlib import Path

from rlearn import RLearn_Model
from rlearn.sports.main_class import TrainAndTestConfig

base_config_path = Path("./content/config/exp_fifawc2022.json")
cached_config_path = Path("./content/config/exp_fifawc2022_with_cached_class_weight.json")
class_weight_cache_path = Path("./content/output/class_weight/exp_fifawc2022_class_weights.pt")

config = json.loads(base_config_path.read_text(encoding="utf-8"))
if "class_weight_fn" not in config:
    raise ValueError(
        "`/content/config/exp_fifawc2022.json` に `class_weight_fn` がありません。"
        "先に `class_weight_fn.type` などを設定してください。"
    )

# Set num_workers to 0 to avoid DataLoader worker issues
config["datamodule"]["num_workers"] = 0

config["class_weight_fn"]["cache_path"] = str(class_weight_cache_path)

cached_config_path.parent.mkdir(parents=True, exist_ok=True)
cached_config_path.write_text(
    json.dumps(config, ensure_ascii=False, indent=4),
    encoding="utf-8",
)

if not class_weight_cache_path.exists():
    raise FileNotFoundError(
        f"Cached class weights not found: {class_weight_cache_path}\n"
        "先に `train_fifawc2022_class_weight_only.py` を実行してください。"
    )

RLearn_Model(
    state_def="PVS"
).run_rlearn(
    run_split_train_test=False,
    run_preprocess_observation=False,
    run_train_and_test=True,
    run_visualize_data=False,
    train_and_test_config=TrainAndTestConfig(
        exp_name="sarsa_attacker",
        run_name="no_class_weight",
        accelerator="gpu",
        devices=1,
        strategy="auto",
        exp_config_path=str(cached_config_path),
        use_class_weights=False,
        output_base_dir="./content/output_no_class_weight",
        class_weight_only=False,
        save_q_values_csv=True,
        max_games_csv=1,
        max_sequences_per_game_csv=5,
    ),
)

print("\n✅ Training completed successfully with cached class weights!")