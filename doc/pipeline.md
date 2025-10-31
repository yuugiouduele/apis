# 05_api_pipeline.md
## 因果構造的ニューラルアーキテクチャ ― 解析APIと運用統合設計

---

### I. 目的

本章では、学習済み因果構造モデルを**クラウド上でAPI化し、定期解析・外部連携可能な形**に統合する設計を定義する。  
対象は以下の3層構成である：

1. **Model Layer** — 学習済みモデル（因果構造・Attention・Utility制御）  
2. **Service Layer** — FastAPIによる解析エンドポイント群  
3. **Pipeline Layer** — Cloud Scheduler / PubSub による定期処理・自動推論  

---

### II. 全体アーキテクチャ構成図

```bash
         ┌──────────────────────────────────┐
         │          Frontend UI             │
         │ (dashboard / visualization)      │
         └──────────────────────────────────┘
                        │ REST/gRPC
                        ▼
         ┌──────────────────────────────────┐
         │          FastAPI Server           │
         │  /predict, /causal_map, /status   │
         └──────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │       Model Inference Core        │
         │ (CausalEncoder + GNN + Attention) │
         └──────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │   Firestore / BigQuery / S3       │
         │ (データベース・モデルストレージ) │
         └──────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │ Cloud Scheduler / PubSub Trigger  │
         │ (定期解析・再学習ジョブ実行)      │
         └──────────────────────────────────┘
```
---

### III. Model Layer 構成

#### 1. モデルロード構造
```python
from models import CausalModel

def load_model():
    model = CausalModel.load_from_checkpoint(
        "./checkpoints/final_model.pt",
        map_location="cuda"
    )
    model.eval()
    return model
2. 推論関数
python
コードをコピーする
def infer(input_data):
    X = preprocess(input_data)
    with torch.no_grad():
        z = model.causal_encoder(X)
        H = model.lattice_encoder(z)
        A = model.causal_attention(H)
        O = model.meta_utility(A)
    Y_hat = output_head(O)
    return postprocess(Y_hat)
3. モデル管理仕様
項目	内容
保存形式	.pt（PyTorch checkpoint）
バージョン管理	Firestoreのmodel_registryコレクション
メタ情報	version, timestamp, train_dataset, seed, metrics

IV. Service Layer（FastAPI）
1. 基本構成
python
コードをコピーする
from fastapi import FastAPI, UploadFile
from model import load_model, infer

app = FastAPI(title="Causal Analysis API", version="1.0.0")
model = load_model()
2. エンドポイント設計
エンドポイント	メソッド	機能	入出力例
/predict	POST	データ入力から因果的予測	JSON: { "X": [...] }
/causal_map	POST	因果Attentionヒートマップ出力	PNG/Base64
/latent	GET	潜在空間zの可視化	.pkl or .json
/status	GET	モデルバージョンと稼働状況	JSON

例: /predict 実装

python
コードをコピーする
@app.post("/predict")
async def predict(data: dict):
    try:
        output = infer(data["X"])
        return {"status": "ok", "prediction": output}
    except Exception as e:
        return {"status": "error", "message": str(e)}
V. Pipeline Layer（クラウド統合）
1. 定期解析フロー
ステップ	内容	実装例
(1) Trigger	Cloud Scheduler (毎日午前3時)	GCP cron job
(2) Fetch	Firestore / S3 から新データ取得	Python client
(3) Inference	モデル推論実行	FastAPI / internal job
(4) Store	結果を Firestore に格納	document update
(5) Notify	Pub/Sub 経由で通知	event-driven

例: スケジューラジョブ

bash
コードをコピーする
gcloud scheduler jobs create http daily-causal-analysis \
  --schedule="0 3 * * *" \
  --uri="https://<API_URL>/predict" \
  --http-method=POST
2. Pub/Sub 構成
Topic: causal-analysis-trigger

Subscriber: pipeline-worker

イベント: 新規データ登録時に自動推論発火

python
コードをコピーする
def callback(event, context):
    data = json.loads(base64.b64decode(event['data']).decode('utf-8'))
    response = requests.post(API_URL + "/predict", json={"X": data["features"]})
VI. Firestore 構造設計
コレクション	フィールド	説明
patients	patient_id, omics_data, histology, label	生体情報
analysis_results	run_id, prediction, causal_map_ref, timestamp	推論結果
model_registry	model_id, metrics, λ_params, version	モデル管理
logs	job_id, latency, status, flops_reduction	運用監視

VII. 運用監視とログ
1. モデル動作ログ
出力：/logs/api/causal_server.log

内容：

受信データ件数

推論時間

GPU使用率（pynvml）

Utility層活性（平均u値）

2. 監視メトリクス
メトリクス	説明	監視間隔
latency(ms)	平均応答時間	5分
throughput	処理件数/分	1分
error_rate	異常応答率	1時間
gpu_util	GPU稼働率	10分

監視は Prometheus + Grafana にて可視化。

VIII. セキュリティ・アクセス制御
機能	方法
認証	Firebase Auth / JWT
アクセス制限	Cloud IAM（read/write分離）
API鍵管理	Secret Manager経由
通信	HTTPS + OAuth2
データ暗号化	AES256 + Firestore暗号化有効化

IX. 拡張統合シナリオ
1. 量子化学計算との連携
モデルAPIが分子構造を入力として因果特徴を算出

外部量子計算API（例: Psi4, Qiskit Chemistry）と連携して、
潜在z空間と電子密度マップの因果対応を構築。

2. Stable Diffusion連動パイプライン
因果特徴 → プロンプト生成API (/generate_prompt)

Stable Diffusionを呼び出し、生成画像をFirestoreに格納

解析結果と画像を統合ダッシュボードに表示

python
コードをコピーする
@app.post("/generate_prompt")
def generate_prompt(data: dict):
    features = extract_causal_features(data)
    prompt = causal_to_prompt(features)
    image = sd_pipeline(prompt)
    save_to_storage(image)
X. デプロイ戦略
環境	サービス	設定例
本番	Cloud Run	GPUインスタンス A100
開発	Docker Compose	local port 8080
バッチ	Vertex AI Pipelines	retraining + evaluation
ストレージ	Firestore + Cloud Storage	データ/モデル永続化

Cloud Run デプロイ例

bash
コードをコピーする
gcloud run deploy causal-api \
  --image gcr.io/<PROJECT_ID>/causal_model:latest \
  --region=asia-northeast1 \
  --memory=8Gi --cpu=4 --gpu=1 --allow-unauthenticated
XI. エラーハンドリング設計
レベル	発生原因	処理方法
モデル層	欠損値, NaN入力	入力バリデーション + 欠損補完
API層	JSONフォーマット不正	400返却 + log記録
Pipeline層	推論失敗	自動リトライ（3回）
Firestore層	書き込み失敗	バックオフ + Pub/Sub再試行
Scheduler層	タイムアウト	次回スケジュールで再実行

XII. 出力例（レスポンス）
json
コードをコピーする
{
  "status": "ok",
  "version": "1.2.5",
  "prediction": {
    "probability": 0.81,
    "class": "cancer_progression"
  },
  "meta": {
    "causal_score": 0.73,
    "utility_entropy": 0.42,
    "timestamp": "2025-10-30T09:00:00Z"
  }
}
XIII. 将来的拡張構想
拡張方向	内容
マルチモーダル統合	画像・オミクス・テキスト因果特徴の同時処理
自動再学習	Cloud Functionsでデータ増加検知→再訓練
ローカルデプロイ	ONNX変換 + Edge GPU向け最適化
解釈支援	Attention + SHAPによる因果的説明性強化
個別患者最適化	Personalized Causal Embedding の動的更新

XIV. 次章
次章 /docs/06_quantum_integration.md では、
量子化学計算および電子密度ベースの潜在因果モデルへの統合設計を定義する。
