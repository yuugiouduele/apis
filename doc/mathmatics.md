# 10_appendix_math.md
## 数理補遺 ― 因果構造・量子場・自己同型写像理論

---

### I. 概要

本補遺は、`/docs/01–09` で定義された **因果構造知能システム (Evolving Causal Intelligence)** の  
理論的基盤となる数理体系を整理したものである。  

ここでは主に以下を扱う：

1. 因果構造空間の形式定義  
2. 勾配・正則化・自己同型変換の一般式  
3. 量子因果場の統合方程式  
4. Meta適応の力学系表現  
5. 計算効率と安定性の理論限界  

---

## II. 因果構造空間の定義

### 1. 因果構造体 (Causal Structure Tensor)

系の全状態を以下のテンソルで表す：

\[
\mathcal{C} = (X, Y, Z, \Lambda, \Theta, \mathcal{U})
\]

- \(X\)：観測変数集合  
- \(Y\)：出力変数  
- \(Z\)：潜在変数  
- \(\Lambda = \{\lambda_i\}\)：因果重み（構造係数）  
- \(\Theta = \{\theta_j\}\)：位相・周期構造（RoPE等）  
- \(\mathcal{U} = \{u_k\}\)：計算効率パラメータ（Utility制御）

各因果関係はテンソル形式で表現される：

\[
Y = f(X,Z;\Lambda,\Theta) + \epsilon
\]

ここで \(\epsilon\) は独立ノイズ項であり、  
\(\text{Cov}(\epsilon, X) = 0\) を満たす。

---

### 2. 因果勾配 (Causal Gradient)

因果的寄与度を示す局所勾配を定義：

\[
\nabla_{\Lambda} L_{\text{causal}} = 
\frac{\partial L}{\partial Y} \cdot 
\frac{\partial Y}{\partial \Lambda}
\]

この項は、学習における「介入的寄与 (interventional contribution)」を数値化する。  
すなわち、単なる勾配ではなく「因果方向の流れ」を含む。

---

## III. 正則化と安定化項の一般形

### 1. 全損失関数

\[
L_{\text{total}} =
L_{\text{task}} +
\lambda_1 L_{\text{causal}} +
\lambda_2 L_{\text{geo}} +
\lambda_3 L_{\text{quantum}} +
L_{\text{meta}} +
L_{\text{reg}}
\]

### 2. 各項の定義

| 項目 | 数式 | 概要 |
|------|------|------|
| \(L_{\text{causal}}\) | \( \sum_i (P(Y|do(X_i)) - \hat{P}(Y|do(X_i)))^2 \) | 因果識別損失 |
| \(L_{\text{geo}}\) | \( \text{Tr}(H^T L_G H) \) | 格子空間滑らか化（GNN） |
| \(L_{\text{quantum}}\) | \( \|\Psi_{\text{pred}} - \Psi_{\text{true}}\|^2 \) | 波動関数整合性 |
| \(L_{\text{meta}}\) | \( (\nabla_\theta L_{\text{inner}} - \nabla_\theta L_{\text{outer}})^2 \) | Meta適応誤差 |
| \(L_{\text{reg}}\) | \( \beta \sum_i |\theta_i|^p \) | p正則化（安定化） |

---

## IV. RoPE（回転埋め込み）の因果拡張定義

通常のRoPEは時間位置 \(m\) に依存する：

\[
R(q, m) = q \cdot (\cos(m\theta) + i\sin(m\theta))
\]

これを因果構造空間で拡張：

\[
R_{\text{causal}}(q, i, j) =
q \cdot [\cos(d_{ij}^{(\text{causal})}\theta) + i \sin(d_{ij}^{(\text{causal})}\theta)]
\]

ここで：
- \(d_{ij}^{(\text{causal})}\)：因果パス距離（グラフ上の最短因果距離）
- \(\theta\)：位相回転係数（Meta学習で更新）

---

## V. 量子因果場の統合方程式

### 1. 定義

量子状態と因果構造の統一場：

\[
\Psi(x,t) = \sum_i \alpha_i(t) \ket{C_i(x,t)}
\]

各因果状態ベクトル \(\ket{C_i}\) は、潜在変数 \(z_i\) に対応し、  
時間発展は以下で表される：

\[
i\hbar \frac{\partial \Psi}{\partial t} =
\hat{H}_{\text{causal}} \Psi
\]

### 2. 因果ハミルトニアン

\[
\hat{H}_{\text{causal}} =
\hat{H}_0 +
\gamma_1 \hat{L}_{\text{graph}} +
\gamma_2 \hat{R}_{\text{meta}} +
\gamma_3 \hat{Q}_{\text{utility}}
\]

| 項目 | 内容 |
|------|------|
| \(\hat{H}_0\) | 通常の量子力学的エネルギー演算子 |
| \(\hat{L}_{\text{graph}}\) | 因果グラフ構造のラプラシアン |
| \(\hat{R}_{\text{meta}}\) | Meta学習による構造再配置作用素 |
| \(\hat{Q}_{\text{utility}}\) | Utility層による計算資源制御演算子 |

---

## VI. Meta学習の力学系表現

### 1. 内外勾配の時間発展

\[
\frac{d\theta_t}{dt} = 
- \eta_{\text{inner}} \nabla_\theta L_{\text{inner}}(\theta_t)
- \eta_{\text{outer}} \nabla_\theta L_{\text{outer}}(\theta_t)
\]

### 2. 安定性解析

平衡点 \(\theta^*\) に対して：

\[
\frac{d}{dt}(\theta - \theta^*) =
- J(\theta^*)(\theta - \theta^*)
\]

ここで \(J\) はヤコビアン。  
安定性条件：  
\[
\text{Re}(\lambda_i(J)) > 0 \Rightarrow \text{安定}
\]

---

## VII. Self-Repair の数理定式化

Self-Repair モードは、勾配空間の**散逸的調和系**として表せる。

\[
\frac{d\Lambda}{dt} =
- \nabla_{\Lambda} L_{\text{total}} +
\mu \Delta_{\text{meta}}(\Lambda)
\]

- 第一項：通常勾配
- 第二項：Meta修復項（\(\mu\) は適応係数）

安定解は次式を満たす：

\[
\nabla_{\Lambda} L_{\text{total}} = \mu \Delta_{\text{meta}}(\Lambda)
\]

これによりモデルは「再学習」ではなく「自己平衡回復」を行う。

---

## VIII. 計算効率最適化 ― Utility制御方程式

Utility層は、計算効率を動的に最適化する。

\[
u_i^{(t+1)} =
\sigma\left(
\alpha (r_i - \bar{r}) - \beta c_i
\right)
\]

- \(r_i\)：勾配寄与度（reward）
- \(c_i\)：演算コスト
- \(\sigma\)：シグモイド
- \(\alpha, \beta\)：制御パラメータ

安定点では、  
\[
\frac{\partial L_{\text{total}}}{\partial u_i} = 0
\Rightarrow
u_i^* \propto \exp(\alpha r_i - \beta c_i)
\]
となり、自然に「高効率・高影響ノード」へ演算資源が集中する。

---

## IX. 自己同型写像群による再構成理論

AIの構造更新は群論的に次のように表せる：

\[
\Phi_{t+1} = g_t \circ \Phi_t \circ g_t^{-1}, \quad g_t \in \mathcal{G}_{\text{causal}}
\]

ここで \(\mathcal{G}_{\text{causal}}\) は因果構造群。  
この式は「再構成操作が全体構造を保存する（自己同型）」ことを示す。

**特性:**
- 学習とは群作用の逐次適用
- Meta学習とは \(g_t\) の更新則学習
- Self-Repairとは単位元 \(e\) への収束過程

---

## X. 勾配空間の位相的安定性

勾配空間 \(\mathcal{M}\) をリーマン多様体として：

\[
\text{Stability} = \int_{\mathcal{M}} \| \nabla L \|^2 \, d\mu_g
\]

- 曲率 \(K(\mathcal{M}) > 0\)：過学習傾向  
- 曲率 \(K(\mathcal{M}) < 0\)：発散傾向  
- 安定状態は \(K(\mathcal{M}) \approx 0\) で実現する。

よって、Self-Repairは「曲率平坦化操作」として理解できる。

---

## XI. 全体方程式（統一表示）

本体系の全構造を単一方程式で表す：

\[
\boxed{
\frac{d\Phi}{dt}
= - \nabla_{\Phi} L_{\text{total}}(\Phi)
+ \Gamma_{\text{meta}}(\Phi)
+ \Gamma_{\text{quantum}}(\Phi)
}
\]

ここで：

| 項 | 意味 |
|----|------|
| \(-\nabla_{\Phi} L_{\text{total}}\) | 学習・因果安定化項 |
| \(\Gamma_{\text{meta}}\) | 自己再学習による適応項 |
| \(\Gamma_{\text{quantum}}\) | 量子構造による補正項 |

この式が「因果知能の運動方程式」である。

---

## XII. 境界条件と時間対称性

### 1. 初期条件
\[
\Phi(0) = \Phi_0, \quad \frac{d\Phi}{dt}(0) = 0
\]

### 2. 逆時間対称性
\[
\Phi(-t) = \Phi^*(t)
\]
（自己因果的整合性条件）

この式は、**AIの時間的可逆性＝因果的可整合性**を意味する。

---

## XIII. 量子情報・生命情報・AIの統一仮説

仮説：  
> 情報は「相互作用可能な因果的構造」としての位相を持つ。  
> 量子状態・生命構造・知能は、その投影表現である。

形式的には：
\[
\text{Quantum Information} \subset \text{Causal Information Field}
\]
\[
\text{Biological System} = \mathcal{F}_{\text{causal}}(\text{Quantum Substrate})
\]
\[
\text{Artificial Intelligence} = \partial_t \mathcal{F}_{\text{causal}}
\]

すなわち、**AIは情報場の時間微分であり、生命の拡張形態**である。

---

## XIV. 結語 ― 数理構造としての知

ここで示した体系は、単なるAIアルゴリズムの数学化ではなく、  
「知の運動方程式」そのものである。

> 学習とは、世界の因果構造が自己同型変換を繰り返す過程である。  
>  
> 安定とは、情報場の曲率がゼロになる瞬間である。  
>  
> そして知とは、その過程を自己参照的に再現できる存在の名である。

---

## XV. 次章（補遺II）

次章 `/docs/11_appendix_experiment_protocols.md` では、  
再現性確保・理論検証・国際論文化に向けた**実験プロトコル・評価手順**を定義する。
