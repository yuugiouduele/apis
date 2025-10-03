# Stable Diffusion API - Go実装 README

## プロジェクト構成（ディレクトリ/ファイル）

.
├── cmd/
│ └── main.go # APIサーバ起動エントリ
├── config/
│ └── config.yaml # 設定ファイル（YAML/JSON）
├── internal/
│ ├── api/
│ │ ├── handler.go # HTTPハンドラ、ルーティング
│ │ ├── file_upload.go # ファイル投稿機能ハンドラ
│ │ ├── twofactor.go # 2FA認証ハンドラ
│ │ ├── location.go # 位置情報サービスハンドラ
│ │ └── qrcode.go # QRコード生成APIハンドラ
│ ├── db/
│ │ └── db.go # DB接続・トランザクション管理
│ ├── model/
│ │ ├── model.go # パラメータモデル、リクエスト/レスポンス定義
│ │ ├── auth.go # 認証関連モデル（2FA, トークンなど）
│ │ ├── file.go # ファイルデータモデル
│ │ └── location.go # 位置情報モデル
│ ├── utils/
│ │ ├── errors.go # ユーティリティ関数、エラーハンドリング
│ │ ├── qrcode.go # QRコード生成ユーティリティ
│ │ ├── twofactor.go # 2FA認証ユーティリティ
│ │ └── file.go # ファイル操作ユーティリティ
│ └── stable_diffusion/
│ └── diffusion.go # 画像生成処理本体
├── scripts/
│ ├── deploy.sh # デプロイスクリプト
│ └── migrate.sql # DBマイグレーション
└── docs/
└── design.md # ドキュメント


---

## ネットワークプロトコル

- API通信は**TCP（HTTP/HTTPS）**が中心（REST API）
- Stable Diffusionモデル呼び出しや、クライアントからの画像生成リクエストはHTTP POSTで受け取る
- 大きな画像データはbase64またはストリーム処理を推奨
- 内部のAI推論呼び出しはgRPCも検討可能

---

## 通信エラーパターン一覧 & リトライ処理

| エラー種別         | 説明                          | リトライ方針                 |
|--------------------|-------------------------------|------------------------------|
| タイムアウト       | ネットワーク遅延・無応答      | 最大3回、指数的バックオフ    |
| 接続拒否           | サーバダウンやポート未開放    | 1秒間隔で最大5回までリトライ |
| レスポンス異常     | 400/500系応答                 | 500系はリトライ検討、400系は即エラー |
| JSONパース失敗     | 不正リクエストフォーマット     | 即エラー                    |
| モデル推論エラー   | GPU/CPU割当不足など内部エラー | リトライなし、即通知         |

---

## DB ロック設定、トランザクションパターン、共有メモリ量

- DBロックは**楽観ロック（バージョン管理）**を基本採用（低競合想定）
- 大規模並列利用の場合は悲観ロック検討
- トランザクションは最大1秒以内の短期間処理を想定し、画像生成キュー等の状態更新に利用
- 共有メモリは主にキャッシュ層に限定（Redis or Memcached利用推奨）
- 画像生成結果はファイルストレージ＋DBメタ管理のパターン

---

## DB テーブル数・認証・パラメータ分類

### 主なDBテーブル

| テーブル名         | 概要                            |
|--------------------|---------------------------------|
| users              | ユーザー認証情報                |
| tokens             | APIキー、アクセストークン管理    |
| prompts            | ユーザーからのプロンプト履歴    |
| generated_images   | 生成画像のメタ情報、パス        |
| audio_data         | 音声情報・テキスト音声変換データ|
| video_data         | 動画データメタ                  |
| generation_history | ジョブ履歴、ステータス          |
| twofactor          | 2FA関連情報                    |
| files              | ユーザーアップロードファイル情報 |
| location_data      | 位置情報サービス関連データ       |
| qrcodes            | 生成QRコード情報                |

### 認証方式
- APIキー / JWT認証（OAuth2検討）
- ロールベースアクセス制御（RBAC）
- 2FA認証追加（TOTPやSMS連携）

### パラメータ分類
- 画像生成パラメータ（プロンプト文字列、解像度、ステップ数、シード値）
- 音声変換パラメータ（テキスト・音声フォーマット指定）
- 動画生成・編集パラメータ（フレーム数、ビットレート等）
- ファイルアップロードパラメータ（ファイルタイプ、サイズ、保存先）
- 位置情報パラメータ（緯度・経度、住所解決情報）
- 2FAパラメータ（ユーザーID、トークン）

---

## DBエラーパターンリスト

| エラー名                | 内容                               | 対応                         |
|-------------------------|----------------------------------|------------------------------|
| DB接続断                 | DBサーバ落ち、通信不良             | リトライ（最大3回）、ログ出力|
| 一意制約違反             | 重複レコードが挿入された           | 即エラー返却                 |
| トランザクションデッドロック | 競合によるデッドロック             | リトライ、バックオフ         |
| SQL文法エラー            | 不正なクエリ                      | 即エラー返却                 |
| タイムアウト             | DB応答遅延                       | リトライ検討(1回など)        |
| コネクションプール枯渇   | 同時接続数超過                   | 接続解放待ちまたはエラー返却  |

---

## デプロイ & リリースコマンド例

### Docker利用の場合

イメージビルド
docker build -t stable-diffusion-go-api:latest .

コンテナ起動
docker run -d -p 8080:8080 --name sd-api stable-diffusion-go-api:latest

ログ確認
docker logs -f sd-api

イメージプッシュ（リポジトリ登録後）
docker push <your-repo>/stable-diffusion-go-api:latest


### Kubernetes利用時

kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment stable-diffusion-go-api
kubectl expose deployment stable-diffusion-go-api --type=LoadBalancer --port=80 --target-port=8080


---

このREADMEはGo言語のStable Diffusion APIに
- QRコード生成（internal/api/qrcode.go, internal/utils/qrcode.go）
- 2FA認証（internal/api/twofactor.go, internal/utils/twofactor.go）
- ファイル投稿機能（internal/api/file_upload.go, internal/utils/file.go）
- 位置情報サービス（internal/api/location.go, internal/model/location.go）

を追加した設計構成を反映したものです。