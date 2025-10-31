# 02_architecture.md
## 因果構造的ニューラルアーキテクチャ ― モジュール設計とデータフロー

---

### I. 目的と設計方針

本章では `/docs/01_theory.md` で定義した理論モデルを、
実装可能なモジュール構造として具体化する。

本アーキテクチャの指針は以下の通り：

1. **因果性を第一級構造として扱う。**  
   各層は単なるデータ変換でなく「因果的写像（Causal Mapping）」である。

2. **幾何学的連結性を明示的に導入する。**  
   データ空間を格子的トポロジーに変換し、Attentionを局所幾何構造に制約する。

3. **学習効率を自己最適化する。**  
   自律計算制御層（Meta Utility Layer）が勾配伝搬の「意味」を評価し、
   無駄な計算を自発的に抑制する。

---

### II. モジュール一覧

| モジュール名 | 目的 | 主な入力/出力 | 実装クラス例 |
|---------------|------|----------------|----------------|
| `CausalEncoder` | 因果パラメータ抽出（潜在因果表現） | X → z | `CausalFNN` |
| `LatticeEncoder` | 因果グラフ構造の格子化とGNN伝播 | z → H_lattice | `CausalGNN` |
| `CausalAttention` | 因果距離に基づくAttention重み付け | H_lattice → A_causal | `CausalTransformer` |
| `MetaUtilityLayer` | 計算資源最適化と報酬制御 | A_causal → O_final | `MetaController` |
| `LossIntegrator` | 各層の勾配と正則化の統合 | 各L → L_total | `LossManager` |

---

### III. 全体フロー

```java
     ┌───────────────────────────┐
     │        Input Data X        │
     └────────────┬──────────────┘
                  ↓
    ┌───────────────────────────┐
    │   1. Causal Encoder (FNN)  │
    │  - 潜在因果変数 z を抽出     │
    └────────────┬──────────────┘
                  ↓
    ┌───────────────────────────┐
    │  2. Lattice Encoder (GNN)  │
    │  - 因果格子構造を生成        │
    └────────────┬──────────────┘
                  ↓
    ┌───────────────────────────┐
    │   3. Causal Attention      │
    │  - RoPEを因果距離に変換     │
    └────────────┬──────────────┘
                  ↓
    ┌───────────────────────────┐
    │ 4. Meta Utility Layer      │
    │ - 計算効率最適化（u_i制御）│
    └────────────┬──────────────┘
                  ↓
    ┌───────────────────────────┐
    │ 5. Loss Integrator         │
    │ - L_task + λL_causal 等    │
    └────────────┬──────────────┘
                  ↓
            ┌──────────┐
            │  Output   │
            └──────────┘
```

---

### IV. 各モジュールの詳細設計

#### 1. CausalEncoder (FNN)
**役割:**  
入力データ X から潜在因果パラメータ z を抽出する。

**構造:**
```python
z = f_theta(X)
Y_hat = g_phi(z, T)
```

内部要素:

標準FNN + LayerNorm

Dropout + causal mask (介入Tの影響除去)

出力層で潜在表現 z を正規化 (z / ||z||)

懸念点:

勾配流が弱い場合、zが無意味化する。

λ_causal により重みを強制的に持たせる。

2. LatticeEncoder (GNN)
役割:
z を格子的トポロジーに投影し、因果的連結を明示化する。

主要式:

𝐻
𝑖
(
𝑙
+
1
)
=
𝜎
(
∑
𝑗
∈
𝑁
(
𝑖
)
1
𝑑
𝑖
𝑑
𝑗
𝑊
(
𝑙
)
𝐻
𝑗
(
𝑙
)
)
H 
i
(l+1)
​
 =σ 
​
  
j∈N(i)
∑
​
  
d 
i
​
 d 
j
​
 
​
 
1
​
 W 
(l)
 H 
j
(l)
​
  
​
 
ここで N(i) は因果隣接ノード集合

𝑑
𝑖
d 
i
​
 ：ノード次数

設計指針:

エッジ重みを因果距離 
𝑑
𝑐
𝑎
𝑢
𝑠
𝑎
𝑙
d 
causal
​
  に基づいてスパース化

ノード更新は勾配安定化のため残差結合

3. CausalAttention (Transformer Block)
役割:
因果的RoPEを組み込み、Attention重みを因果的連結に変換。

式:

𝐴
𝑖
𝑗
=
Softmax
(
𝑄
𝑖
⋅
𝐾
𝑗
⊤
𝑑
+
𝛾
𝑓
(
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
)
)
A 
ij
​
 =Softmax( 
d
​
 
Q 
i
​
 ⋅K 
j
⊤
​
 
​
 +γf(d 
causal
​
 (i,j)))
RoPE拡張:

𝑄
𝑖
,
𝐾
𝑗
=
RoPE
𝑐
𝑎
𝑢
𝑠
𝑎
𝑙
(
𝑄
𝑖
,
𝐾
𝑗
,
𝑑
𝑐
𝑎
𝑢
𝑠
𝑎
𝑙
)
Q 
i
​
 ,K 
j
​
 =RoPE 
causal
​
 (Q 
i
​
 ,K 
j
​
 ,d 
causal
​
 )
通常のトークン位置 
𝑚
m → 因果距離 
𝑑
𝑐
𝑎
𝑢
𝑠
𝑎
𝑙
d 
causal
​
  に置換

θ関数を学習可能パラメータ化

出力:

A_causal：因果Attention表現

H_out：残差接続により安定化

4. MetaUtilityLayer
役割:
演算ブロックごとのutilityを動的制御。
計算量と損失改善度に基づいて勾配流を抑制/促進。

式:

𝑢
𝑖
=
𝜎
(
𝛼
Δ
𝐿
𝑖
−
𝛽
𝐶
𝑖
)
u 
i
​
 =σ(αΔL 
i
​
 −βC 
i
​
 )
高い 
𝑢
𝑖
u 
i
​
 ：計算を実行

低い 
𝑢
𝑖
u 
i
​
 ：skip connection活性

実装案:

```python
for block in model.blocks:
    delta_L = estimate_loss_reduction(block)
    u = torch.sigmoid(alpha * delta_L - beta * block.cost)
    if u > threshold:
        y = block(y)
```
注意点:

ループ内でbackpropを通すため、uを微分可能に保つ。

コスト正規化が重要（C_iをGPU FLOPSで近似可）

5. LossIntegrator
役割:
各損失を統合して最終的な L_total を計算。

𝐿
𝑡
𝑜
𝑡
𝑎
𝑙
=
𝐿
𝑡
𝑎
𝑠
𝑘
+
𝜆
1
𝐿
𝑐
𝑎
𝑢
𝑠
𝑎
𝑙
+
𝜆
2
𝐿
𝑔
𝑒
𝑜
𝑚
𝑒
𝑡
𝑟
𝑖
𝑐
+
𝜆
𝑟
𝑒
𝑔
𝐿
𝑟
𝑒
𝑔
L 
total
​
 =L 
task
​
 +λ 
1
​
 L 
causal
​
 +λ 
2
​
 L 
geometric
​
 +λ 
reg
​
 L 
reg
​
 
サブ構成:

L_task: CrossEntropy / MSE / CosineSim

L_causal: P(Y|do(T))安定性損失

L_geo: ノード連結安定性損失

L_reg: L_DC + L_Dis + L_Stable

V. 学習ループ設計（擬似コード）
```python
for epoch in range(EPOCHS):
    for X, Y, T in dataloader:
        # 1. forward
        z = causal_encoder(X)
        H = lattice_encoder(z)
        A = causal_attention(H, T)
        O = meta_utility(A)
        Y_hat = output_head(O)

        # 2. loss
        L_task = task_loss(Y_hat, Y)
        L_causal = causal_loss(Y_hat, Y, T)
        L_geo = geometric_loss(H)
        L_reg = regularization(z, Y)
        L_total = L_task + λ1*L_causal + λ2*L_geo + L_reg

        # 3. backward
        optimizer.zero_grad()
        L_total.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```
VI. 安定化および再現性管理
再現性確保

```python
torch.manual_seed(42)
```

乱数・データシャッフル固定

Docker環境化で依存性固定

学習安定化

Gradient Clipping (1.0)

Dynamic λ scheduler

Warmup + CosineAnnealing LR

BatchNormで勾配爆発回避

VII. 計算複雑度（概算）
部分	オーダー	コメント
FNN (CausalEncoder)	O(N·d²)	標準線形変換
GNN (LatticeEncoder)	O(E·d)	スパース化により軽量化
Attention (Causal)	O(N²·d)	causal距離近傍のみ採用で削減
MetaUtilityLayer	O(B)	ブロック単位制御
Total	≈ O(N log N)（近似）	実用レベルで安定

VIII. 出力仕様
O_final: 因果的特徴統合表現（N×d）

L_total: 損失総和

u: 各ブロックのutilityベクトル

logs: 勾配・λ・安定性メトリクス（tensorboard可視化対象）

IX. 次章
次章 /docs/03_training.md では、
学習スケジュール、ハイパーパラメータ、再現性管理、
および評価基準の詳細を記述する。

