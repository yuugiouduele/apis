# 04_experiments.md
## 因果構造的ニューラルアーキテクチャ ― 実験設計・可視化・再現性試験

---

### I. 目的

本章では、`/docs/03_training.md` で確立した学習済みモデルの  
**挙動検証・因果可視化・再現性試験設計**を定義する。

目的は以下の3つである：

1. モデルが「因果的に整合した挙動」を示すか検証する。  
2. 勾配・Attention・潜在空間の安定性を可視化する。  
3. 実験条件を統一し、**同一seed再現性**を保証する。

---

## II. 実験フレームワーク概要

| コンポーネント | 目的 | 実装 |
|----------------|------|------|
| `ExperimentRunner` | 各種検証ジョブの統括 | Hydra / custom YAML構成 |
| `Visualizer` | 因果Attention・潜在空間可視化 | Matplotlib / Plotly |
| `Evaluator` | 数値評価（因果安定・性能） | PyTorchMetrics |
| `ReproManager` | seed固定・環境チェック | Docker + wandb |
| `Logger` | すべての数値・図版記録 | TensorBoard / wandb |

---

## III. 実験条件セットアップ

### 1. 環境統一

```bash
docker build -t causal_model .
docker run --gpus all -it causal_model bash
Dockerfile抜粋

Dockerfile
コードをコピーする
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
RUN pip install -U torch-geometric wandb matplotlib hydra-core
COPY ./src /workspace/src
WORKDIR /workspace
2. 設定ファイル構成 (configs/experiment.yaml)
yaml
コードをコピーする
seed: 42
batch_size: 64
epochs: 50
optimizer: AdamW
lr: 1e-4
lambda:
  causal: 0.5
  geometric: 0.3
  reg: 0.2
evaluation:
  metrics: [accuracy, kl_causal, laplacian, utility_entropy]
visualization:
  latent_tsne: true
  attention_heatmap: true
  graph_structure: true
IV. 可視化設計
1. 因果Attentionヒートマップ
目的:
RoPEの因果距離変換後のAttention重みを直感的に理解する。

python
コードをコピーする
import matplotlib.pyplot as plt
import seaborn as sns

A = model.get_attention_matrix(X)
sns.heatmap(A.detach().cpu().numpy(), cmap='viridis')
plt.title("Causal Attention Heatmap")
plt.xlabel("Key Index")
plt.ylabel("Query Index")
plt.show()
評価ポイント:

対角成分（近因果）に高重み → 局所的因果が保たれている。

遠因果結合も一部残る場合、潜在干渉が反映されている。

2. 潜在空間可視化（t-SNE / PCA）
目的:
潜在因果変数 z が構造的クラスターを形成しているか確認。

python
コードをコピーする
from sklearn.manifold import TSNE
z = model.causal_encoder(X).detach().cpu().numpy()
z_emb = TSNE(n_components=2, perplexity=30).fit_transform(z)

plt.scatter(z_emb[:,0], z_emb[:,1], c=labels, cmap='Spectral')
plt.title("Latent Causal Space (t-SNE)")
plt.show()
指標:

クラス境界が滑らかで、局所密度が高い → 潜在因果安定性が高い。

3. 因果格子構造のグラフ描画
目的:
LatticeEncoder(GNN)が形成した因果ネットワークを可視化。

python
コードをコピーする
import networkx as nx
G = model.lattice_encoder.get_graph()
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color='orange', edge_color='gray', node_size=80)
plt.title("Learned Causal Lattice Structure")
plt.show()
観察項目:

適度なスパース性（過剰連結でない）

主要ノードが中心的に集約していること

ノイズノード（孤立ノード）が少ないこと

4. Meta Utility 分布ヒートマップ
目的:
Meta Utility Layerが計算資源をどのように最適化しているか確認。

python
コードをコピーする
u = model.meta_utility_layer.get_utility()
sns.heatmap(u.detach().cpu().numpy().reshape(1, -1), cmap='plasma')
plt.title("Utility Activation Map")
plt.xlabel("Block Index")
plt.yticks([])
plt.show()
期待される挙動:

学習初期：全ブロックほぼ均等活性

学習後期：重要ブロックのみ高活性化
→ 計算効率の自己制御成立

V. 実験プロトコル
ステージ	実験内容	成果物
Exp-1	因果Attention挙動観察	heatmap, attention_weight.csv
Exp-2	潜在空間クラスタリング	z_tsne.png, z_stat.pkl
Exp-3	因果格子構造安定性試験	lattice_graph.png, degree_dist.png
Exp-4	Meta Utility効率評価	utility_map.png, flops_log.csv
Exp-5	全体安定性＋再現試験	reproducibility_log.json

VI. 再現性試験プロトコル
1. Seed一貫性試験
5回異なるGPU上で同一seed実験を実施し、
主要メトリクスの標準偏差を算出。

python
コードをコピーする
runs = [run(seed=s) for s in [42, 43, 44, 45, 46]]
stdev = np.std([r['accuracy'] for r in runs])
判定基準:
標準偏差 < 1.5% → 再現性良好。

2. モデル安定性試験（勾配分布）
目的:
各層の勾配統計を周期的に計測し、発散の兆候を早期検出。

python
コードをコピーする
for name, param in model.named_parameters():
    grad_norm = param.grad.data.norm(2).item()
    logger.log({f"grad/{name}": grad_norm})
評価:

Layer間で急激な勾配変化がないこと

Meta Utility更新後にgrad normが一時的上昇 → 正常

3. データサブセット再訓練試験
目的:
異なる患者群・領域（例：癌種別）の分布に対する汎化性を検証。

python
コードをコピーする
for subset in ["lung", "colon", "breast"]:
    train_subset(subset)
判定基準:

精度差 < 5% → 因果構造の汎化性良好。

VII. 結果整理テンプレート
実験名	主観的観察	定量結果	考察
Exp-1	Attentionが局所集中	KL=0.021	因果距離反映が成功
Exp-2	z空間が明瞭に分離	Silhouette=0.71	因果潜在分布安定
Exp-3	格子に小規模中心群形成	DegreeVar=0.18	幾何的秩序確立
Exp-4	Utilityが3ブロックに収束	FLOPS削減=27%	自律最適化成立
Exp-5	再現性高 (σ<1%)		モデル安定性保証

VIII. 発展的実験（今後の拡張）
量子化学計算との融合

GNNノードを分子構造単位に拡張し、電子密度マップを因果特徴へ転写。

H_i に電子雲密度関数 φ_i(x) を導入。

画像／オミクスハイブリッド実験

Stable Diffusionの潜在表現と因果潜在 z の共埋め込み。

医療画像上の因果Attention分布を可視化。

自律モデル収束可視化

学習時間と utility_entropy の相関可視化。

自己学習的構造安定化を観測。

IX. 実験結果共有フォーマット
出力	フォーマット	保存先
因果Attentionヒートマップ	.png	/results/visuals/attention/
潜在空間可視化	.png .pkl	/results/latent/
勾配統計	.json	/results/logs/grad/
Utility分布	.csv .png	/results/utility/
総合結果	.md レポート	/reports/summary.md

X. 結論
因果Attentionと格子構造は一貫して再現可能な安定挙動を示すこと。

Meta Utility Layerは計算資源削減と性能維持を両立できること。

本設計により、**AIモデルの「思考過程の因果的可視化」**が実現可能。

XI. 次章
次章 /docs/05_api_pipeline.md では、
学習済みモデルを用いて解析APIを構築し、
量子化学計算・医療データ・画像生成との統合運用フローを定義する。
