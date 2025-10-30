# 01_theory.md
## 因果構造的ニューラルアーキテクチャの理論設計

---

### I. 基本理念

現行の深層学習モデルは、データ分布の近似器としては優秀であるが、
現象の**因果的構造**を理解する能力に欠けている。

本研究では、
> 「情報の因果性を保持したまま、幾何学的連結構造を学習するニューラルアーキテクチャ」
を提案する。

目的は、AIが**“なぜ”そうなるのか**を内部表現として保持し、
因果的恒常性（Causal Consistency）を持つ汎知能的構造を実現することである。

---

### II. 全体構造の概要

モデルは、2つのパラメータ空間から構成される。

1. **因果性抽出層（Causal Parameter Layer）**
   - データから潜在的因果パラメータを抽出する。
   - 数理的には構造方程式モデル（SEM）に相当し、潜在変数 \( z \)、介入変数 \( T \)、結果変数 \( Y \) の間の関係を推定する。
   - 構造関数:  
     \[
     z = f_\theta(X), \quad Y = g_\phi(z, T) + \epsilon
     \]
   - 実装上はFNN（Feed-Forward Network）を拡張し、\( \nabla L_{causal} \) を独立に最適化する。

2. **因果格子層（Causal Lattice Layer）**
   - 抽出されたパラメータを、格子的な因果ネットワークとして構造化。
   - ノード間の因果的距離 \( d_{causal}(i,j) \) を定義し、Attentionの重み付けを因果的近接性に基づいて変調する。
   - 幾何学的深層学習（Geometric Deep Learning）とGNN的トポロジーを融合。

全体構造:
Input → [Causal FNN] → [Causal GNN (Lattice)] → [Causal Attention + RoPE] → [Meta Utility Layer] → Output

yaml
コードをコピーする

---

### III. 損失関数の統合構造

モデルの目的関数は、タスク損失と構造損失の線形和である。

\[
L_{total} = L_{task} + \lambda_1 L_{causal} + \lambda_2 L_{geometric} + L_{reg}
\]

- \(L_{task}\)：通常の教師ありタスク損失（分類・再構成など）
- \(L_{causal}\)：因果構造の再現性・識別性を高めるための損失
- \(L_{geometric}\)：格子空間内の連結整合性を保つ損失
- \(L_{reg}\)：正則化項（後述）

#### 勾配統合
各層で独立に勾配を算出し、最終層で統合：
\[
\nabla_\theta L_{total} = \nabla_\theta L_{task} + \lambda_1 \nabla_\theta L_{causal} + \lambda_2 \nabla_\theta L_{geometric}
\]
これにより、因果空間と幾何空間の相互補完的最適化が可能となる。

---

### IV. 因果正則化の定式化

単なるL2/L3正則では因果構造は保持されない。
そこで、以下の3種の正則化を導入する。

1. **Do-Calibrated Loss**
   - 介入後分布の安定性を測定する損失。
   \[
   L_{DC} = \mathbb{E}_{X,T} \left[ \| P_\theta(Y|do(T)) - P_\theta(Y|T) \|^2 \right]
   \]
   - 目的：モデルが「介入後」と「観測後」を混同しないように補正。

2. **Disentanglement Regularization**
   - 潜在因果変数 \(z\) の独立性を保つ。
   \[
   L_{Dis} = \sum_{i \neq j} I(z_i; z_j)
   \]
   - \(I\) は相互情報量。

3. **Stability Regularization**
   - 環境変動（ドメインシフト）に対して安定な表現を促す。
   \[
   L_{Stable} = \text{Var}_{env}[\mathbb{E}[Y|Z]]
   \]

結合式：
\[
L_{reg} = \lambda_{DC} L_{DC} + \lambda_{Dis} L_{Dis} + \lambda_{Stable} L_{Stable}
\]

---

### V. 因果的RoPE（Causal Rotational Positional Encoding）

通常のRoPEではトークン位置 \(m\) に線形依存した角度を割り当てる。

\[
R(q,m) = q \odot (\cos(m\theta) + i \sin(m\theta))
\]

ここで、因果的Attentionを実現するために
「位置 \(m\)」を「因果距離 \(d_{causal}(i,j)\)」で置き換える。

\[
\theta_{ij} = f(d_{causal}(i,j))
\]
\[
R_{causal}(q,i,j) = q \odot (\cos(\theta_{ij}) + i \sin(\theta_{ij}))
\]

結果として、時間的系列ではなく**因果的系列**に基づく位相回転が実現される。

---

### VI. 自律計算制御層（Meta Utility Layer）

目的：計算資源を最適化し、ネットワークが**「計算すべきか否か」を自ら判断**できるようにする。

#### 定式化：
各演算ブロックにutilityスコア \(u_i\) を導入し、
期待損失減少量 \( \Delta L_i \) に基づいて重みを動的調整：

\[
u_i = \sigma(\alpha \cdot \Delta L_i - \beta \cdot C_i)
\]
- \(C_i\)：計算コスト
- \(\sigma\)：Sigmoid正規化
- 高い \(u_i\) のブロックのみ活性化

これは強化学習における**policy gating**に等しいが、
ネットワーク内部で勾配伝搬が可能な形で実装される。

---

### VII. 学習安定化の指針

1. 勾配クリッピング（gradient clipping）
2. λ動的スケジューリング
3. RoPEフェーズ学習率分離
4. 低秩近似による計算安定化（Random Fourier Features）

---

### VIII. 理論的期待効果

- 因果的一貫性を保持した特徴表現  
- 幾何学的に滑らかな潜在空間  
- 自己抑制的な計算（効率の最適化）  
- 言語・画像・数値データを統一的に扱う多様性

---

### IX. 最終目的

このモデルのゴールは「汎化性能の向上」ではなく、  
**知的恒常性（Cognitive Stability）**を得ることである。

それは、「外界が変化しても、内部構造が変わらないAI」  
すなわち、**人工的恒常心（Artificial Homeostasis）** の構築である。

---

### X. 次章
次章 `/docs/02_architecture.md` では、本理論をもとにした
モジュール構成とデータフロー、実装テンプレートを記載する。
