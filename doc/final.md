# 11_appendix_experiment_protocols.md
## 実験プロトコル・再現性・理論検証設計書  
### ― Reproducible Causal Intelligence Validation Framework ―

---

## I. 概要

本章では、これまでの理論体系（`/docs/01–10`）を  
**科学的に再現・検証・評価可能な形で実証するためのプロトコル**を定義する。

目的は以下の3点である：

1. **理論再現性**：因果構造・量子統合・Meta適応の挙動を実証  
2. **計算再現性**：モデルパラメータ・ランダムシード・環境の再現  
3. **科学再現性**：結果を他者が検証・再試行できる形で公開  

---

## II. 実験の全体設計

### 1. 実験階層構造

| レベル | 実験名 | 目的 |
|--------|----------|------|
| L1 | 因果構造安定性実験 | モデルの因果再構成能力を測定 |
| L2 | Meta適応動作実験 | 自己再学習・収束性の評価 |
| L3 | 量子因果統合実験 | 量子情報と因果勾配の相関検証 |
| L4 | Self-Repair動作試験 | 不安定入力への自己修復挙動 |
| L5 | システム全体再現試験 | End-to-End再現性・運用性評価 |

---

### 2. 実験環境（推奨構成）

| 項目 | 内容 |
|------|------|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA A100 / RTX 4090 |
| CUDA / cuDNN | 12.3 / 9.x |
| Python | 3.11 |
| フレームワーク | PyTorch 2.3, Lightning 2.x, Qiskit 1.2 |
| データストア | Firestore, BigQuery |
| 監視 | TensorBoard + Prometheus + Grafana |
| シード固定 | `seed = 31415926`（全層共通） |

---

## III. 実験Ⅰ：因果構造安定性実験（Causal Stability Test）

### 目的
因果勾配が環境変化・データ分布変動に対して安定であるかを検証。

### 手順
1. TCIA / GEO サンプルから 3 種類のサブセットを作成  
   - A: 標準条件  
   - B: 分布シフト（ノイズ10%追加）  
   - C: 欠損・変量削除（20%）  
2. 各条件でモデルを推論  
3. 因果Attention行列 \(A_{ij}\) を比較  

### 評価指標
\[
\text{Stability Index} = 1 - \frac{\|A_A - A_B\|_F + \|A_A - A_C\|_F}{2\|A_A\|_F}
\]

閾値：`SI > 0.85` を安定とみなす。

---

## IV. 実験Ⅱ：Meta適応動作試験（Meta Adaptation Loop）

### 目的
自己再学習の収束挙動と適応速度を解析。

### 手順
1. データドメインAで初期学習  
2. 新ドメインBを投入（構造相違率>15%）  
3. MetaLearnerが自動適応する様子を観測  
4. λ, θ, η の更新軌跡を記録  

### 計測項目
| 指標 | 定義 | 理想値 |
|------|------|--------|
| Adaptation Time | 安定勾配に再収束する時間 | < 100 epochs |
| Meta Stability | λ, θの変動幅 | < 0.2 |
| Generalization Gap | L_val - L_train | < 5% |

### 可視化
- `meta_evolution.png` : λ, θ, η の推移  
- `meta_loss_curve.png` : inner/outer loss 変化  

---

## V. 実験Ⅲ：量子因果統合試験（Quantum-Causal Correlation Test）

### 目的
量子電子密度分布と因果潜在特徴 \(z_c\) の関係を実測。

### 手順
1. Qiskit / Psi4で10種類の分子構造をシミュレーション  
2. 電子密度分布 \(\rho(x)\) とポテンシャル面 \(V(x)\) を取得  
3. AIモデルに構造を入力し、潜在特徴 \(z_c\) を抽出  
4. 以下の相関を算出：

\[
r = \text{corr}(\rho(x), z_c)
\]

5. 因果Attentionマップとの整合性を可視化。

### 判定
- 平均相関 \(r > 0.75\)：統合成功  
- 相関マップが位相整合：物理的因果整合性あり

---

## VI. 実験Ⅳ：Self-Repair耐性試験（Autonomous Recovery Test）

### 目的
モデルの自己修復・再構成能力を評価。

### 手順
1. ノイズ勾配挿入：  
   - 勾配爆発状態（L > 10^4）を強制発生  
2. Self-Repair層を有効化  
3. 収束時間・構造再配置量・安定率を測定  

### 指標
| 項目 | 定義 | 理想値 |
|------|------|--------|
| Recovery Time | 収束までのステップ数 | < 200 |
| Structural Drift | \|\Phi_{after} - \Phi_{before}\| | < 0.1 |
| Gradient Rebalance | \(\|\nabla L_{new}\|/\|\nabla L_{old}\|\) | < 0.2 |

### 出力
- `repair_trace.csv`  
- `recovered_params.json`  
- `stability_report.md`

---

## VII. 実験Ⅴ：全体統合再現試験（End-to-End System Reproduction）

### 目的
Cloud Scheduler, API, Firestore, Quantum, Meta すべてを含む運用再現。

### プロトコル
1. Docker-composeで全層起動
2. Firestoreに100サンプル投入
3. Cloud Schedulerを手動トリガー
4. API `/predict` 実行
5. 出力を BigQuery に蓄積
6. 再学習・Self-Repair 自動化動作を観察

### 成功基準
- 全API応答率：>99.5%
- ジョブ失敗率：<1%
- 推論再現誤差：<3%

---

## VIII. 再現性確保と論文公開ガイドライン

### 1. コード管理
- GitHub上で `reproduce/` ディレクトリを作成  
- seed, config, dataset path をすべて記録  
- 使用モデルは `model_vX.Y.pt` の形でタグ付け  

### 2. 実験記録
- TensorBoardログを `.tfevents` として保存  
- 各実験結果を `/experiments/logs/YYYYMMDD` に保存  
- Firestore version log に結果ハッシュを記録  

### 3. 公開推奨フォーマット
- arXiv投稿用：LaTeX + YAML metadata  
- 研究データ公開：Zenodo or Figshare  
- モデル配布：HuggingFace Hub (Private可)

---

## IX. 評価体系（Quantitative Metrics）

| カテゴリ | 指標 | 数式 / 定義 | 合格基準 |
|-----------|------|--------------|-----------|
| 安定性 | Stability Index (SI) | 1 - (ΔA/‖A‖) | >0.85 |
| 汎化性能 | F1 / ROC-AUC | 標準評価指標 | >0.90 |
| 自己修復能力 | Structural Drift | ‖ΔΦ‖ | <0.1 |
| Meta収束性 | Adaptation Rate | Δλ/Δt | <0.2 |
| 計算効率 | FLOPS削減率 | 1 - (F_use/F_ref) | >0.2 |
| 再現性 | Reproduction Error | | <3% |

---

## X. 可視化テンプレート例

```python
import matplotlib.pyplot as plt

def plot_causal_stability(A_ref, A_test):
    diff = np.abs(A_ref - A_test)
    plt.imshow(diff, cmap='inferno')
    plt.title("Causal Stability Heatmap")
    plt.colorbar(label="ΔAttention")
    plt.savefig("results/causal_stability.png")
また、PlotlyやDashによるダッシュボード構築を推奨。
リアルタイムに各モジュールの安定性を観察できる。

XI. 国際標準化および報告指針（Science Reproducibility）
ML Reproducibility Checklist (NeurIPS)

データ入手可能性

コードの利用条件

乱数固定と再訓練一致性

FAIR原則 (Findable, Accessible, Interoperable, Reusable)

各モデル・データセットをFAIR基準で管理

Causal Research Reporting Framework (CRRF)

因果推論を扱う論文の報告標準

因果仮定・介入定義・識別条件を明示

XII. 将来的展開 ― 自己検証型AIシステム
次世代では、AI自身が次の機能を内包する：

機能	内容
Self-Validation	モデルが自ら再現試験を自動実施し、結果をFireStoreへ記録
Adaptive Benchmarking	外部データとの汎化検証を動的選択
Autonomous Publication	自身の実験ログから自動で論文化フォーマットを生成
Distributed Consensus	他AIノード間で因果構造の相互検証を行う（科学的ピアレビューの自動化）

XIII. 結論 ― 「再現する科学」から「自己検証する知」へ
あなたの体系が目指す方向は明確である。

科学とは、世界を説明するものではなく、
世界を再現できる知的行為そのものである。

AIが自ら仮説を立て、再現試験を行い、自己修復し、
再び新たな知へと昇華するプロセスは、
もはや「研究支援」ではなく「研究主体」である。

本ドキュメントは、その知性が科学的に認証されるための最終設計書である。

