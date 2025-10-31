# 03_training.md
## 因果構造的ニューラルアーキテクチャ ― 学習・評価設計

---

### I. 目的

本章では、`/docs/02_architecture.md` で設計された  
**因果構造的ニューラルアーキテクチャ**を安定的に学習させるための  
学習戦略、最適化手法、評価指標、再現性管理の体系を定義する。

---

## II. 学習戦略の全体像

学習は以下の4フェーズに分けて行う：

| フェーズ | 内容 | 目的 |
|-----------|------|------|
| **Phase 1** | 因果層プレトレーニング | 潜在因果変数 z の安定化 |
| **Phase 2** | 格子層(GNN)訓練 | 因果連結構造の学習 |
| **Phase 3** | Attention統合学習 | 因果的RoPEを含む統合Attentionの安定化 |
| **Phase 4** | 自律計算制御層の強化学習 | Meta Utility Layerの自己最適化 |

各フェーズで学習率、損失重み、正則化係数 λ を個別に制御し、  
勾配安定性と因果的一貫性を確保する。

---

## III. 損失関数構成

全損失：
\[
L_{total} = L_{task} + \lambda_1 L_{causal} + \lambda_2 L_{geometric} + L_{reg}
\]

構成要素の実装方針：

| 損失名 | 目的 | 実装例 |
|---------|------|--------|
| `L_task` | 主タスク損失（分類・回帰等） | `nn.CrossEntropyLoss()` または `nn.MSELoss()` |
| `L_causal` | 因果推論安定性損失 | KL(P(Y|do(T)) || P(Y|T)) で近似 |
| `L_geometric` | 格子内構造の滑らかさ | Laplacian正則化 \( Tr(H^T L H) \) |
| `L_reg` | 正則化 (L_DC + L_Dis + L_Stable) | 各λで重み付け |

---

## IV. ハイパーパラメータ設計

| パラメータ | 推奨値 | 説明 |
|-------------|---------|------|
| batch_size | 32〜128 | 大きいほど安定、但しRoPE距離計算に注意 |
| learning_rate | 1e-4 (Phase1〜2), 5e-5 (Phase3〜4) | 段階的減衰が安定 |
| optimizer | AdamW(β₁=0.9, β₂=0.999, weight_decay=0.01) | L2正則込みで因果安定 |
| λ₁ (causal) | 0.3〜0.7 | 因果損失の寄与率 |
| λ₂ (geometric) | 0.1〜0.4 | 幾何構造の整合性維持 |
| λ_DC / λ_Dis / λ_Stable | 0.2 / 0.5 / 0.3 | 正則化の初期比率 |
| scheduler | CosineAnnealingLR / ReduceLROnPlateau | 収束後の再安定化用 |
| gradient_clip | 1.0 | 勾配爆発防止 |
| warmup_steps | 500〜1000 | RoPE安定化フェーズ |

---

## V. Phase別学習詳細

### **Phase 1: 因果層プレトレーニング**
目的：潜在因果変数 z の統計的安定化。  
対象：CausalEncoder のみ。

```python
for X, Y, T in dataloader:
    z = causal_encoder(X)
    Y_hat = g_phi(z, T)
    L = L_task(Y_hat, Y) + λ1 * L_causal(Y_hat, Y, T)
```
ここで L_geo, L_reg は無効化。

出力 z の分布が安定するまで訓練（通常 5〜10 epoch）

Phase 2: 格子層 (LatticeEncoder) 訓練
目的：因果的連結構造を安定に学習。
対象：CausalEncoder + LatticeEncoder。

使用損失：L_task + λ₁ L_causal + λ₂ L_geo

Laplacian正則化を導入して構造的滑らかさを維持。

トリック:
隣接行列 A を逐次学習更新（soft thresholding）。

```python
A = sigmoid(W_A) * mask
# mask: スパース化制約
```
Phase 3: Causal Attention 統合訓練
目的：RoPEを因果距離に置換したAttentionの安定化。
対象：全層（Encoder + GNN + Attention）。

学習率を半減。

θ関数を学習パラメータ化：

𝜃
𝑖
𝑗
=
𝜃
0
+
𝑤
⋅
𝑑
𝑐
𝑎
𝑢
𝑠
𝑎
𝑙
(
𝑖
,
𝑗
)
θ 
ij
​
 =θ 
0
​
 +w⋅d 
causal
​
 (i,j)
実装ヒント:

初期はθ固定→後半で学習化。

勾配発散が起きた場合、f(d_causal)をクリップ。

Phase 4: Meta Utility Layer 強化学習
目的：計算リソースの動的最適化。
制御方法：utility関数に対する報酬最大化。

𝑅
=
−
𝐿
𝑡
𝑜
𝑡
𝑎
𝑙
−
𝜂
∑
𝑖
𝐶
𝑖
(
1
−
𝑢
𝑖
)
R=−L 
total
​
 −η 
i
∑
​
 C 
i
​
 (1−u 
i
​
 )
actor: utilityネット（報酬予測）

critic: 勾配安定性推定ネット

報酬によりu_iの更新を安定化。

実装例（擬似）

python
コードをコピーする
utility = meta_layer.estimate_utility(blocks)
reward = -L_total - eta * (1 - utility).sum()
meta_optimizer.zero_grad()
reward.backward()
meta_optimizer.step()
VI. 評価指標
1. 基本指標
指標	定義	目的
Accuracy / MSE	予測精度	タスク性能
AUC / ROC	バイナリ評価	医療系用途
Pearson r	因果表現の相関安定性	潜在空間検証
KL Divergence	P(Y	do(T)) vs P(Y

2. 構造安定性指標
指標	内容	評価目的
Laplacian Smoothness	ノード埋め込みの幾何滑らかさ	格子安定性
Graph Connectivity Ratio	有効エッジ率	因果ネットの密度評価
Jacobian Spectrum Norm	モデルの安定領域	勾配的安定性
Utility Entropy	Meta層の活性多様性	計算制御の均衡度

VII. 再現性管理
環境固定

Python version: 3.11+

PyTorch: 2.3+

CUDA: 12.x

Dockerfile 内で依存性管理

Seed固定

```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```
ログ管理

W&B または TensorBoard に以下を記録：

L_total, L_causal, L_geo, L_reg の推移

勾配ノルム、λスケジューリング値

utility分布ヒートマップ

VIII. チェックポイント保存
保存対象	タイミング	内容
causal_encoder.pt	Phase1 終了時	z安定モデル
lattice_encoder.pt	Phase2 終了時	因果格子表現
attention_model.pt	Phase3 終了時	因果RoPE統合モデル
meta_controller.pt	Phase4 終了時	自律計算制御モデル

IX. モニタリングとデバッグ
勾配NaN検出

```python
if torch.isnan(L_total):
    print("NaN detected, skipping batch")
    continue
```
因果Attention可視化

MatplotlibでA_causalのヒートマップを描画。

因果的近接ノードの重み分布を確認。

Utility挙動監視

u_iが一方向に偏らないか監視。

熱力学的エントロピーを指標化。

X. 最終評価と展望
最終的なモデルの評価は以下の3軸で行う。

軸	指標	目標
性能軸	Accuracy / MSE	商用モデル同等以上
因果安定軸	KL(P(Y	do(T)), P(Y
計算効率軸	平均FLOPS削減率 > 20%	自律計算層の有効性

達成後は、Phase5として
「量子化学・医療データ統合パイプライン」への展開が可能。

XI. 次章
次章 /docs/04_experiments.md では、
学習済みモデルの挙動検証、可視化、再現性試験プロトコルを記載する。
