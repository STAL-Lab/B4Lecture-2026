from rlearn import RLearn_Model
from pathlib import Path

# デバッグのために、RLearn_Modelに渡される前にゲームIDが検出されるか確認します。
input_data_path = Path("./data/fifawc/preprocess_data/")

game_ids = []
if input_data_path.exists():
    for item in input_data_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            game_ids.append(item.name)

print(f"Detected game IDs: {game_ids}")

# もしゲームIDが検出されていなければ、ここで処理を停止できます。
if not game_ids:
    print("エラー: 検出されたゲームIDがありません。データを分割できません。")
else:
    RLearn_Model(
        state_def="PVS",
        input_path=str(input_data_path),
        output_path="./data/fifawc/preprocess_data/split/",
    ).run_rlearn(
        run_split_train_test=True,
        run_preprocess_observation=False,
        run_train_and_test=False,
        run_visualize_data=False
    )

    print("\n✅ Spliting data completed successfully!")