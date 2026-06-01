# ex4: 前処理の変更と既製モデルによる試合分析

## 概要

本レポートでは、Wyscout イベントデータの前処理を変更して La Liga（Spain）のデータを残し、特定の1試合を選んで配布済み NMSTPP モデルで推論を行い、その結果を可視化・分析した内容をまとめる。

---

## 課題1：前処理の変更

### 残したリーグ

- **La Liga (Spain)** — 2017/18 シーズン

### データダウンロードセルの変更点

元のセルは England 以外のすべてのリーグを削除している。以下の2点を変更した。

- `events_Spain.json` と `matches_Spain.json` の `rm` をコメントアウト（Spain を残す）
- `events_England.json` と `matches_England.json` の `rm` を新たに追加（England を削除）

```python
# Spain (La Liga) は残すので削除しない
# subprocess.run(['rm', '-rf', 'event/events_Spain.json'])
# subprocess.run(['rm', '-rf', 'matches/matches_Spain.json'])

# England (Premier League) を削除（Spainのみ残すため、課題1の変更点）
subprocess.run(['rm', '-rf', 'event/events_England.json'])
subprocess.run(['rm', '-rf', 'matches/matches_England.json'])
```

`preprocess_method="UIED"` で再前処理を実行した結果、生成された `data.csv` の概要は以下の通り。

| 項目 | 値 |
|---|---|
| 試合数 | 380 |
| イベント数（行数）| 549,357 |
| カラム数 | 22 |
| 必須列の欠損 | 0 |

必須列（`action`, `start_x`, `start_y`, `seconds`, `delta_T`, `team`, `match_id`）はすべて存在し、欠損もなかった。

---

## 課題2：1試合のピックアップ

### 選んだ試合

- **match_id**: `2565711`
- **対戦カード**: Real Madrid vs Barcelona
- **公式記録**: Real Madrid 0 - 3 Barcelona

### 選定理由

- La Liga の中で有名な対戦カードで、結果も分かりやすかったため
- 0-3 とスコアが大きく動いた試合のほうがモデルの予測傾向を観察しやすそうだと判断
- 1試合あたりのイベント数も多く（1564件）、分析に十分なデータ量があった

### 備考：UIED 変換後のデータと公式記録の食い違い

UIED 前処理を経た `data.csv` を確認したところ、`home_score` の最大値が `2`、`away_score` の最大値が `0` となっており、公式記録の 0-3 と一致しなかった。`goal=1` となっているイベントもデータ上は 2 件しかなく、3点目に対応するイベントが見当たらなかった。

また `home_team` フラグについても、`home_team=1` となるチームは `team_id=676`、`home_team=0` となるチームは `team_id=675` だったが、`matches_Spain.json` 上では `team_id=675` が `side=home`、`team_id=676` が `side=away` となっており、`data.csv` の `home_team` の対応が逆になっている。

原因は完全には特定できなかったが、UIED 形式への変換時に一部のイベントが落ちている、または `home_team` のセマンティクスが生 Wyscout データの `side` と異なる可能性が考えられる。本課題では試合選択の問題ではなく、前処理の挙動として注意点として記録しておく。

---

## 課題3：配布済みモデルによる推論

### 推論設定

- モデル: NMSTPP
- 学習済み重み: `_model_23.pth`（配布済み）
- 設定ファイル: `hyperparameters.json`（配布済み）
- 正規化辞書: `min_max_dict.json`
- 推論入力: 選んだ試合のイベントのみを抽出した CSV（`match_for_inference.csv`）

課題の指示通り、学習は行わない。

なお、配布モデルの平均的な性能と比較するための参考として、`test.csv`（train/valid/test 分割の test セット、38試合）に対する推論も補助的に実施した。提出ファイルとしては選んだ試合の `loss_df_clasico.csv`（主役）と、参考として test 38試合の `loss_df.csv` の両方を含める。

### loss_df の比較

提出ファイルは `loss_df_clasico.csv`（選んだ試合のみ、主役）と `loss_df.csv`（test 38試合、参考）の両方。両者を並べて示す。

| メトリック | 選んだ試合（主）| test 38試合（参考）|
|---|---|---|
| train_loss | 4.5694 | 4.9958 |
| CEL_action | 0.8936 | 1.0343 |
| ACC_action | **0.7222** | 0.6564 |
| F1_action | 0.1746 | 0.1659 |
| MAE_deltaT | 3.0311 | 3.3013 |
| MAE_x | 7.96 | 8.357 |
| MAE_y | 16.47 | 16.80 |

選んだ試合のほうがすべてのメトリックで誤差が小さく、精度が高い。配布済みモデルにとって、この試合は test セットの平均より予測しやすかったと言える。

---

## 課題4：可視化分析

### 作成した図

1. アクション種別の頻度 — `figures/01_action_frequency.png`
2. ピッチ上のヒートマップ（チーム別）— `figures/02_pitch_heatmap.png`
3. 予測 vs 実アクションの分布比較 — `figures/03_pred_vs_actual.png`
4. HPUS プロット — `figures/HPUS.png`
5. HPUS+ プロット — `figures/HPUS_plus.png`

### アクション種別の頻度

選んだ試合の 1564 イベントの内訳を見ると、`short_pass` が 954 件（61.7%）で圧倒的に多く、続いて possession 境界マーカー `_`（285）、`dribble`（186）と続く。`shot`（29）、`cross`（16）といった頻度の低いアクションはごく少数だった。

### ピッチ上のヒートマップ

`start_x`, `start_y` の分布を hexbin でチームごとに描画した。両チームともピッチ全体に広く分布しており、特定エリアに極端な集中はない。チームごとに細かいホットスポットの位置に違いはあるものの、大まかな分布パターンは似ていた。

### 予測 vs 実アクションの分布

実データと予測結果のアクション分布を比較した結果、配布済み NMSTPP モデルがこの試合で出力した予測アクションは **3 種類のインデックス**（0, 3, 4）のみで、しかも **80.3%** が同じ1つのインデックスに集中していた。出力分布を見ると、最頻インデックス（インデックス 0）は実データの `short_pass` と対応していそうだった。

一方、実データには 8 種類のアクション（`period_over`、`game_over` を除く）が含まれていて、`short_pass` 自体の比率は **61.7%** である。

つまり、表面上の `ACC_action = 0.7222` という数字は、「常に `short_pass` と予測する」だけのトリビアルなベースラインの正解率（≈ 62%）をわずかに上回るだけと解釈できる。本試合における配布済みモデルは、`shot`, `cross`, `carry`, `high_pass`, `long_pass` といった頻度の低いアクションをほぼ予測しておらず、多数派クラスをまねるだけになっている。

### HPUS / HPUS+

HPUS プロットを見ると、前半は両チームとも HPUS のピークが交互に出現し、45 分付近で `team_id=675`（公式の Home 側）が **HPUS ≈ 27** のピークを示している。後半に入ると `team_id=676`（公式の Away 側）の HPUS が徐々に高くなり、**80 分頃に HPUS ≈ 35** という試合中最大のピークに達した。

HPUS+（攻撃に繋がった possession に限定したスコア）ではこの違いがより明確で、`team_id=675` は 65 分以降ほぼ 0 のまま推移する一方、`team_id=676` は 80 分付近で **HPUS+ ≈ 22** のスパイクを示している。

公式記録では `team_id=676`（Barcelona）が 0-3 で勝利しており、後半に得点を重ねた試合だったとされる。HPUS / HPUS+ の動きはこの試合展開と整合しているように見えた。前述のとおり UIED のイベントストリームでは 3 点目に対応するイベントが見当たらなかったが、HPUS による分析では試合の流れを再現できているのが興味深かった。

---

## まとめ：考察

- 配布済みモデルはこの試合で `ACC_action ≈ 0.72` を達成したが、これは実データの `short_pass` 比率が高いことに大きく依存している。実際、予測の約 80% は1つのインデックスに集中しており、出力に使われたインデックスは 3 種類のみだった。
- モデルは多数派アクション（`short_pass`）に予測を寄せる傾向があり、`shot` や `cross` のような頻度の低いアクションをこの試合では区別できていない。
- 選んだ試合は test セット 38 試合の平均よりすべての loss メトリックで誤差が小さく、モデルにとって予測しやすい試合だった。`short_pass` の比率が特に高いことが要因の一つと考えられる。
- HPUS / HPUS+ では、前半・後半でチームごとのスコアの強弱が変化し、公式の試合結果と整合する動きが確認できた。
- 一方、UIED 前処理後の `data.csv` ではスコアと公式記録に食い違いがあり、`home_team` の対応も `matches_Spain.json` と逆になっていた。後続の分析・モデリングではこの点を留意する必要がある。

---

## 参考文献

- Wyscout Open Access Dataset — https://www.nature.com/articles/s41597-019-0247-7
- OpenSTARLab Event Package — https://github.com/open-starlab
- NMSTPP (Yeung & Fujii) — https://github.com/calvinyeungck/Football-Match-Event-Forecast
