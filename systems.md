1. API 仕様案

以下は主要機能に対する API 群の一案です。REST+JSON 形式を想定。認証付き API は JWT トークン等を期待。

各 API 名称、パス、HTTP メソッド、リクエスト・レスポンスの型を示します。
```ts
認証 / 認可 API
機能	メソッド	パス	リクエスト型	レスポンス型	備考
ユーザ登録	POST	/api/auth/register	{ username: string; email: string; password: string; }	{ user: { userId: number; username: string; email: string; }; token: string; refreshToken: string; }	登録と同時にログイン状態にするケース
ログイン	POST	/api/auth/login	{ username: string; password: string; }	{ user: { userId: number; username: string; email: string; }; token: string; refreshToken: string; }	JWT 発行など
トークン更新	POST	/api/auth/refresh	{ refreshToken: string; }	{ token: string; refreshToken: string; }	アクセストークン更新
ログアウト / トークン失効	POST	/api/auth/logout	{ refreshToken: string; }	{ success: boolean; }	リフレッシュトークン失効など
2FA 有効化準備	GET	/api/auth/2fa/setup	—（認証済み）	{ secret: string; otpAuthUri: string; qrCodeImage: string; }	TOTP シークレット生成等
2FA 有効化完了	POST	/api/auth/2fa/enable	{ otpCode: string; }	{ success: boolean; }	ユーザが入力した OTP コードチェック
2FA 無効化	POST	/api/auth/2fa/disable	{ otpCode: string; }	{ success: boolean; }	
PIN 設定	POST	/api/auth/pin/set	{ pin: string; }	{ success: boolean; }	PIN ハッシュ保存
PIN 認証（検証）	POST	/api/auth/pin/verify	{ pin: string; }	{ success: boolean; }	認証補助用
決済認証（OTP / PIN）	POST	/api/auth/payment-auth	`{ paymentId: string; method: "2fa"	"pin"; code: string }`	{ success: boolean; detail?: string }
API トークン発行	POST	/api/auth/api-token	{ scope: string[]; expiresIn: number; }	{ token: string; expiresAt: string; }	アプリ連携用
API トークン一覧取得	GET	/api/auth/api-token	—	{ tokens: { tokenId: number; scope: string[]; expiresAt: string; revoked: boolean }[] }	認証済ユーザ用
API トークン失効	POST	/api/auth/api-token/revoke	{ tokenId: number; }	{ success: boolean; }	
メディア系 API
機能	メソッド	パス	リクエスト型	レスポンス型	備考
動画アップロード開始	POST	/api/media/videos/init-upload	{ filename: string; contentType: string; size: number; }	{ uploadUrl: string; videoId: number; }	署名付き URL 等を返す
動画アップロード完了通知	POST	/api/media/videos/complete-upload	{ videoId: number; }	{ success: boolean; }	ストレージ遷移や処理トリガー
動画メタデータ取得	GET	/api/media/videos/{videoId}	—	{ video: { videoId: number; userId: number; title: string; description: string; duration: number; thumbnailUrl: string; } }	
動画リスト取得	GET	/api/media/videos	?userId=&limit=&offset=	{ videos: { videoId:number; title:string; thumbnailUrl:string; }[] }	ページング対応
字幕追加	POST	/api/media/videos/{videoId}/subtitle	{ languageCode: string; subtitleFilePath: string; }	{ success: boolean; }	
キャプチャログ取得	GET	/api/media/videos/{videoId}/captures	—	{ captures: { timestamp: number; imageUrl: string; }[] }	
音声解析結果取得	GET	/api/media/videos/{videoId}/audio-analysis	—	{ analyses: { segmentIndex: number; features: object }[] }	
字幕翻訳リクエスト	POST	/api/media/videos/{videoId}/translate-subtitle	{ srcLang: string; dstLang: string; }	`{ translationId: number; status: "pending"	"done"
字幕翻訳結果取得	GET	/api/media/videos/{videoId}/translate-subtitle/{translationId}	—	{ status: string; path?: string; }	
AI / 画像生成系 API
機能	メソッド	パス	リクエスト型	レスポンス型	備考
画像生成リクエスト	POST	/api/ai/generate-image	{ prompt: string; modelId?: number; options?: object; }	`{ requestId: string; status: "pending"	"done"
画像生成結果取得	GET	/api/ai/generate-image/{requestId}	—	{ status: string; imageUrl?: string; }	
プロンプト一覧取得	GET	/api/ai/prompts	?userId=&limit=&offset=	{ prompts: { promptId:number; text:string; createdAt:string }[] }	
類似プロンプト取得	GET	/api/ai/prompts/{promptId}/similar	?limit=	{ similarities: { relatedPromptId:number; score: number }[] }	
モデル一覧取得	GET	/api/ai/models	—	{ models: { modelId:number; name:string; version:string; description:string }[] }	
ドメイン / 金融 / 生物系 API（代表例）
機能	メソッド	パス	リクエスト型	レスポンス型	備考
金融価格時系列取得	GET	/api/finance/price	?symbol=string&start=timestamp&end=timestamp	{ prices: { timestamp: string; open: number; high: number; low: number; close: number; volume: number }[] }	
掲示板投稿	POST	/api/forum/posts	{ title: string; body: string; }	{ postId: number; }	
掲示板コメント	POST	/api/forum/posts/{postId}/comments	{ body: string; }	{ commentId: number; }	
仮想座標登録	POST	/api/space/coordinate	{ scale: number; x: number; y: number; z: number; metadata?: object; }	{ coordinateId: number; }	
生物解析結果登録	POST	/api/bio/experiments/{experimentId}/results	{ summary: object; filePath: string; }	{ resultId: number; }	
共通レスポンス例 型定義（TypeScript 風）
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: { code: string; message: string; }
}
```

例：

interface LoginResponse {
  user: { userId: number; username: string; email: string; };
  token: string;
  refreshToken: string;
}

2. ER 図（テーブル関係図の説明＋簡易図）

以下に主要テーブル間のリレーション構造をテキストで説明し、その後に簡易的な “ASCII 図” を載せます。

関係の説明

user_account を中心として、認証系テーブル (user_2fa, user_pin, api_token) はすべて user_id を外部キーとする

media.video_metadata は user_account と関連

media.video_subtitle, video_capture_log, audio_analysis は video_metadata と関連

ai.prompt は user_account と関連

ai.prompt_similarity は prompt 同士の関連を持つ（自己参照的リレーション）

ログ系は user_account との関連を持つ

ドメイン系（例：finance_price, forum_post, space_coordinate, bio_experiment, bio_analysis_result）も user_account または該当テーブルと関連

簡易 ASCII ER 図

[user_account]───┬── (1) ──[user_2fa]
                 │
                 ├──(1)──[user_pin]
                 │
                 └──(1)──[api_token]  

[user_account]───┬──(1)──[video_metadata]───┬──(1)──[video_subtitle]
                 │                         ├──(1)──[video_capture_log]
                 │                         └──(1)──[audio_analysis]

[user_account]───┬──(1)──[prompt]───┬──(many)──[prompt_similarity]
                 │                  └──(many)──[prompt_similarity]

[user_account]───┬──(many)──[audit_log]
                 └──(many)──[access_log]

[user_account]───┬──(many)──[forum_post]───(many)──[forum_comment]
                 │
                 └──(many)──[space_coordinate]

[user_account]───┬──(many)──[bio_experiment]───(many)──[bio_analysis_result]

[finance_price] -- no direct to user (or via symbol-metadata table)


“(1)” は主キー側 1 対 “多” 側を示す

prompt_similarity は中間テーブル的関係を持つ自己参照構造

finance_price は直接ユーザと紐づけない（一部アプリではユーザごとのウォッチ銘柄と紐づけてもよい）

例えば、Mermaid 形式で簡易図を書くと：

```
erDiagram
  user_account ||--o{ user_2fa : has
  user_account ||--o{ user_pin : has
  user_account ||--o{ api_token : issues
  user_account ||--o{ video_metadata : owns
  video_metadata ||--o{ video_subtitle : has
  video_metadata ||--o{ video_capture_log : has
  video_metadata ||--o{ audio_analysis : has
  user_account ||--o{ prompt : owns
  prompt ||--o{ prompt_similarity : related_to
  user_account ||--o{ audit_log : has
  user_account ||--o{ access_log : has
  user_account ||--o{ forum_post : writes
  forum_post ||--o{ forum_comment : receives
  user_account ||--o{ space_coordinate : owns
  user_account ||--o{ bio_experiment : owns
  bio_experiment ||--o{ bio_analysis_result : has
  finance_price : {
    symbol VARCHAR
    timestamp TIMESTAMP
    open DOUBLE
    high DOUBLE
    low DOUBLE
    close DOUBLE
    volume BIGINT
  }
```

このような ERD を README や設計資料に含めておくと、ドメイン構造が見やすくなります。

3. 仕様書統合 Markdown テンプレート

以下は、これまでの API 仕様、ER 図、テーブル設計、アーキテクチャ概要などを順序よくまとめた SPECIFICATION.md（または DESIGN.md 等）テンプレートです。

適宜章・節を追加・編集してご利用ください。

# システム設計仕様書

## 1. 概要

このドキュメントは、本プロジェクト（多機能メディア／AI／解析プラットフォーム）のシステム設計仕様を整理したものです。  
対象読者：開発者、アーキテクト、テスター、運用者など。

主な内容：アーキテクチャ概要、データベース設計（ER 図とテーブル定義）、API 仕様、運用／可観測性設計など。

---

## 2. 技術スタック & アーキテクチャ

### 2.1 技術スタック

- フロントエンド：React + TypeScript  
- バックエンド / API：Go  
- AI / モデル処理層：Python  
- データストア：PostgreSQL, Redis, Qdrant  
- 監視 / ロギング：Prometheus, Grafana, ログ集約ツール  
- Orchestration：Kubernetes  
- CI/CD：GitHub Actions  

### 2.2 全体アーキテクチャ


```
GitHub Actions → (CI/CD) → Kubernetes クラスタ
├→ API サービス（Go）
├→ AI モデルサービス（Python）
├→ フロントエンド（React）
├→ ステージング / テスト環境
└→ 監視スタック（Prometheus, Grafana, ログ集約）
```

- サービス間の認証・認可ゲートウェイを設置  
- 冗長構成、スケール設計、可観測性導入  
- モニタリング／トレース／ログ収集／アラート体制整備  

---

## 3. データベース設計

### 3.1 ER 図（関係図）

以下は主要テーブル間の関係を示した簡易 ER 図（Mermaid 形式も併記）：

```sh
erDiagram
  user_account ||--o{ user_2fa : has
  user_account ||--o{ user_pin : has
  user_account ||--o{ api_token : issues
  user_account ||--o{ video_metadata : owns
  video_metadata ||--o{ video_subtitle : has
  video_metadata ||--o{ video_capture_log : has
  video_metadata ||--o{ audio_analysis : has
  user_account ||--o{ prompt : owns
  prompt ||--o{ prompt_similarity : related_to
  user_account ||--o{ audit_log : has
  user_account ||--o{ access_log : has
  user_account ||--o{ forum_post : writes
  forum_post ||--o{ forum_comment : receives
  user_account ||--o{ space_coordinate : owns
  user_account ||--o{ bio_experiment : owns
  bio_experiment ||--o{ bio_analysis_result : has
  finance_price : {
    symbol VARCHAR
    timestamp TIMESTAMP
    open DOUBLE
    high DOUBLE
    low DOUBLE
    close DOUBLE
    volume BIGINT
  }
```

3.2 主要テーブル定義例

（前段で示した「主要テーブル設計例（型付き）」をここに貼り付け）

認証系
user_account, user_2fa, user_pin, api_token, payment_auth_log

メディア系
video_metadata, video_subtitle, video_capture_log, audio_analysis, subtitle_translation_log

AI モデル / プロンプト系
model, model_parameter, prompt, prompt_similarity

ログ / 履歴系
audit_log, access_log

ドメイン系
finance_price, forum_post, forum_comment, space_coordinate, bio_experiment, bio_analysis_result

3.3 インデックス & 性能設計留意点

検索条件・結合条件に基づくインデックス設計

複合インデックス、部分インデックス、式インデックス、BRIN など使用

JSONB を使うなら、検索対象キーは通常列化

ログ / 時系列データのパーティショニング／アーカイブ設計

リードレプリカ、キャッシュ（Redis）、シャーディング戦略

4. API 仕様
4.1 共通仕様

リクエスト / レスポンス形式：JSON

認証付き API：Authorization ヘッダーに Bearer <token> を含める

共通レスポンス型：

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: { code: string; message: string; }
}

4.2 認証 / 認可 API 一覧

（前段「API 仕様案」の表をここに貼り付け）

4.3 メディア系 API

（同様に、メディア関連 API 表をここに貼り付け）

4.4 AI / 画像生成系 API

（AI / 画像系 API 表をここに貼り付け）

4.5 ドメイン / 解析 / 生物系 API

（ドメイン系 API 表をここに貼り付け）

5. 運用 / モニタリング / セキュリティ設計

GitHub Actions による CI → テスト → デプロイ

ブルー / グリーンデプロイ、カナリア方式、ロールバック設計

Prometheus / Grafana によるメトリクス監視

ログ集約（Loki / ELK / Fluentd 等）＋検索・可視化

分散トレーシング (OpenTelemetry / Jaeger 等)

アラート閾値設計、異常検出、通知 (Slack, PagerDuty)

サービス間の認証・認可設計（内部 API のアクセス制御）

鍵・シークレット管理（Kubernetes Secrets, Vault, KMS 等）

復旧・フェイルオーバー設計、バックアップ／リストア体制

6. 拡張可能性・今後の課題

認証方式拡張 (WebAuthn / FIDO2)

モデルレジストリ、バージョニング基盤

OLAP ライヤー導入 (ClickHouse, Apache Druid 等)

マルチリージョン展開、シャーディング・水平分割戦略

高可用性構成、障害自動回復設計

性能テスト・負荷試験設計


# システム設計仕様書

## 1. 概要

このドキュメントは、本プロジェクト（多機能メディア／AI／解析プラットフォーム）のシステム設計仕様を整理したものです。  
対象読者：開発者、アーキテクト、テスター、運用者など。  

主な内容：アーキテクチャ概要、データベース設計（ER 図とテーブル定義）、API 仕様、運用／可観測性設計など。

---

## 2. 技術スタック & アーキテクチャ

### 2.1 技術スタック

- フロントエンド：React + TypeScript  
- バックエンド / API：Go  
- AI / モデル処理層：Python  
- データストア：PostgreSQL, Redis, Qdrant  
- 監視 / ロギング：Prometheus, Grafana, ログ集約ツール  
- コンテナ / Orchestration：Kubernetes  
- CI/CD：GitHub Actions（ビルド・テスト・デプロイの自動化）  

### 2.2 アーキテクチャ概観

```
GitHub Actions → (CI/CD) → Kubernetes クラスタ
├→ API サービス（Go）
├→ AI モデルサービス（Python）
├→ フロントエンド（React）
├→ ステージング / テスト環境
└→ 監視スタック（Prometheus, Grafana, ログ集約）
```
- 各サービス間通信には認証／認可ゲートウェイを設置  
- 冗長構成・スケール設計・可観測性導入  
- メトリクス・ログ・トレースの一元収集とアラート体制を整備  

---

## 3. データベース設計

### 3.1 ER 図（関係図）

以下は主要テーブル間の関係を示した簡易 ER 図（Mermaid 形式）：

```mermaid
erDiagram
  user_account ||--o{ user_2fa : has
  user_account ||--o{ user_pin : has
  user_account ||--o{ api_token : issues
  user_account ||--o{ video_metadata : owns
  video_metadata ||--o{ video_subtitle : has
  video_metadata ||--o{ video_capture_log : has
  video_metadata ||--o{ audio_analysis : has
  user_account ||--o{ prompt : owns
  prompt ||--o{ prompt_similarity : related_to
  user_account ||--o{ audit_log : has
  user_account ||--o{ access_log : has
  user_account ||--o{ forum_post : writes
  forum_post ||--o{ forum_comment : receives
  user_account ||--o{ space_coordinate : owns
  user_account ||--o{ bio_experiment : owns
  bio_experiment ||--o{ bio_analysis_result : has
  finance_price : {
    symbol VARCHAR
    timestamp TIMESTAMP
    open DOUBLE
    high DOUBLE
    low DOUBLE
    close DOUBLE
    volume BIGINT
  }
```

3.2 主要テーブル定義例（型付き）
```sh
認証 / ユーザ関係
CREATE TABLE auth.user_account (
  user_id BIGSERIAL PRIMARY KEY,
  username VARCHAR(100) NOT NULL UNIQUE,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE auth.user_2fa (
  user_id BIGINT PRIMARY KEY REFERENCES auth.user_account(user_id) ON DELETE CASCADE,
  totp_secret TEXT,
  is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
  backup_codes TEXT[],  
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE auth.user_pin (
  user_id BIGINT PRIMARY KEY REFERENCES auth.user_account(user_id) ON DELETE CASCADE,
  pin_hash VARCHAR(255) NOT NULL,
  is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
  attempt_count INT NOT NULL DEFAULT 0,
  last_attempt_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE auth.api_token (
  token_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES auth.user_account(user_id) ON DELETE CASCADE,
  token VARCHAR(512) NOT NULL UNIQUE,
  token_type VARCHAR(50) NOT NULL,
  scope TEXT[],
  issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  revoked BOOLEAN NOT NULL DEFAULT FALSE,
  last_used_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE auth.payment_auth_log (
  log_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES auth.user_account(user_id),
  payment_id VARCHAR(100),
  auth_method VARCHAR(50),
  status VARCHAR(50),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  metadata JSONB
);

メディア / 動画関係
CREATE TABLE media.video_metadata (
  video_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES auth.user_account(user_id),
  title VARCHAR(255) NOT NULL,
  description TEXT,
  upload_path TEXT NOT NULL,
  thumbnail_path TEXT,
  duration_seconds INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE media.video_subtitle (
  subtitle_id BIGSERIAL PRIMARY KEY,
  video_id BIGINT NOT NULL REFERENCES media.video_metadata(video_id) ON DELETE CASCADE,
  language_code VARCHAR(10) NOT NULL,
  subtitle_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (video_id, language_code)
);

CREATE TABLE media.video_capture_log (
  capture_id BIGSERIAL PRIMARY KEY,
  video_id BIGINT NOT NULL REFERENCES media.video_metadata(video_id),
  timestamp_seconds INT NOT NULL,
  capture_image_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE media.audio_analysis (
  analysis_id BIGSERIAL PRIMARY KEY,
  video_id BIGINT NOT NULL REFERENCES media.video_metadata(video_id),
  segment_index INT NOT NULL,
  features JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE media.subtitle_translation_log (
  id BIGSERIAL PRIMARY KEY,
  video_id BIGINT NOT NULL REFERENCES media.video_metadata(video_id),
  src_language VARCHAR(10) NOT NULL,
  dst_language VARCHAR(10) NOT NULL,
  translation_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

AI モデル / プロンプト系
CREATE TABLE ai.model (
  model_id BIGSERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  version VARCHAR(50) NOT NULL,
  description TEXT,
  metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (name, version)
);

CREATE TABLE ai.model_parameter (
  param_id BIGSERIAL PRIMARY KEY,
  model_id BIGINT NOT NULL REFERENCES ai.model(model_id) ON DELETE CASCADE,
  param_path TEXT NOT NULL,
  checksum VARCHAR(128),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE ai.prompt (
  prompt_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT REFERENCES auth.user_account(user_id),
  text_prompt TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE ai.prompt_similarity (
  id BIGSERIAL PRIMARY KEY,
  prompt_id BIGINT NOT NULL REFERENCES ai.prompt(prompt_id) ON DELETE CASCADE,
  related_prompt_id BIGINT NOT NULL REFERENCES ai.prompt(prompt_id),
  similarity_score DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (prompt_id, related_prompt_id)
);

ログ / 履歴系
CREATE TABLE history.audit_log (
  log_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT REFERENCES auth.user_account(user_id),
  action VARCHAR(100) NOT NULL,
  target_entity VARCHAR(100),
  target_id BIGINT,
  details JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE history.access_log (
  log_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT REFERENCES auth.user_account(user_id),
  ip_address INET,
  user_agent TEXT,
  path VARCHAR(255) NOT NULL,
  status_code INT NOT NULL,
  response_time_ms INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ドメイン / 解析 / 生物系（例）
CREATE TABLE domain.finance_price (
  symbol VARCHAR(50) NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  open_price DOUBLE PRECISION,
  high_price DOUBLE PRECISION,
  low_price DOUBLE PRECISION,
  close_price DOUBLE PRECISION,
  volume BIGINT,
  PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE domain.forum_post (
  post_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT REFERENCES auth.user_account(user_id),
  title TEXT,
  body TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE domain.forum_comment (
  comment_id BIGSERIAL PRIMARY KEY,
  post_id BIGINT NOT NULL REFERENCES domain.forum_post(post_id) ON DELETE CASCADE,
  user_id BIGINT REFERENCES auth.user_account(user_id),
  body TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE domain.space_coordinate (
  coordinate_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT REFERENCES auth.user_account(user_id),
  scale_level INT NOT NULL,
  x_coord DOUBLE PRECISION,
  y_coord DOUBLE PRECISION,
  z_coord DOUBLE PRECISION,
  metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE domain.bio_experiment (
  experiment_id BIGSERIAL PRIMARY KEY,
  user_id BIGINT REFERENCES auth.user_account(user_id),
  name VARCHAR(255),
  description TEXT,
  experiment_date DATE,
  metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE domain.bio_analysis_result (
  result_id BIGSERIAL PRIMARY KEY,
  experiment_id BIGINT NOT NULL REFERENCES domain.bio_experiment(experiment_id) ON DELETE CASCADE,
  result_summary JSONB,
  result_file_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

3.3 インデックス & 性能設計留意点

検索条件・結合条件を元にインデックス設計

複合インデックス、部分インデックス (partial index)、式インデックスの活用

JSONB 列は柔軟性は高いが、検索キーが多いなら正規化を検討

履歴／ログ系データにはパーティショニングやアーカイブ戦略

リードレプリカ構成、キャッシュ（Redis）、シャーディング戦略を想定

4. API 仕様
4.1 共通仕様

リクエスト/レスポンス形式：JSON

認証付き API：Authorization: Bearer <token> ヘッダーを要求

レスポンス共通型（TypeScript 風）：

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: { code: string; message: string; }
}

4.2 認証 / 認可 API
```ts
機能	メソッド	パス	リクエスト型	レスポンス型	備考
ユーザ登録	POST	/api/auth/register	{ username: string; email: string; password: string; }	{ user: { userId: number; username: string; email: string; }; token: string; refreshToken: string; }	新規登録 & ログイン同時化
ログイン	POST	/api/auth/login	{ username: string; password: string; }	{ user: { userId: number; username: string; email: string; }; token: string; refreshToken: string; }	JWT 発行
トークン更新	POST	/api/auth/refresh	{ refreshToken: string; }	{ token: string; refreshToken: string; }	アクセストークン更新
ログアウト / 失効	POST	/api/auth/logout	{ refreshToken: string; }	{ success: boolean; }	リフレッシュトークン無効化
2FA 設定準備	GET	/api/auth/2fa/setup	—（認証済み）	{ secret: string; otpAuthUri: string; qrCodeImage: string; }	TOTP シークレット発行
2FA 有効化	POST	/api/auth/2fa/enable	{ otpCode: string; }	{ success: boolean; }	OTP コード検証
2FA 無効化	POST	/api/auth/2fa/disable	{ otpCode: string; }	{ success: boolean; }	
PIN 設定	POST	/api/auth/pin/set	{ pin: string; }	{ success: boolean; }	PIN 保存
PIN 認証	POST	/api/auth/pin/verify	{ pin: string; }	{ success: boolean; }	認証補助
決済認証	POST	/api/auth/payment-auth	`{ paymentId: string; method: "2fa"	"pin"; code: string }`	{ success: boolean; detail?: string }
API トークン発行	POST	/api/auth/api-token	{ scope: string[]; expiresIn: number; }	{ token: string; expiresAt: string; }	アプリ連携用トークン
トークン一覧取得	GET	/api/auth/api-token	—	{ tokens: { tokenId: number; scope: string[]; expiresAt: string; revoked: boolean }[] }	
トークン失効	POST	/api/auth/api-token/revoke	{ tokenId: number; }	{ success: boolean; }	
4.3 メディア系 API
機能	メソッド	パス	リクエスト型	レスポンス型	備考
動画アップロード開始	POST	/api/media/videos/init-upload	{ filename: string; contentType: string; size: number; }	{ uploadUrl: string; videoId: number; }	署名付き URL 発行
動画アップロード完了通知	POST	/api/media/videos/complete-upload	{ videoId: number; }	{ success: boolean; }	処理開始トリガー
動画メタデータ取得	GET	/api/media/videos/{videoId}	—	{ video: { videoId: number; userId: number; title: string; description: string; duration: number; thumbnailUrl: string; } }	
動画リスト取得	GET	/api/media/videos	?userId=&limit=&offset=	{ videos: { videoId:number; title:string; thumbnailUrl:string; }[] }	ページング対応
字幕追加	POST	/api/media/videos/{videoId}/subtitle	{ languageCode: string; subtitleFilePath: string; }	{ success: boolean; }	
キャプチャログ取得	GET	/api/media/videos/{videoId}/captures	—	{ captures: { timestamp: number; imageUrl: string; }[] }	
音声解析結果取得	GET	/api/media/videos/{videoId}/audio-analysis	—	{ analyses: { segmentIndex: number; features: object }[] }	
字幕翻訳リクエスト	POST	/api/media/videos/{videoId}/translate-subtitle	{ srcLang: string; dstLang: string; }	`{ translationId: number; status: "pending"	"done"
字幕翻訳結果取得	GET	/api/media/videos/{videoId}/translate-subtitle/{translationId}	—	{ status: string; path?: string; }	
4.4 AI / 画像生成系 API
機能	メソッド	パス	リクエスト型	レスポンス型	備考
画像生成リクエスト	POST	/api/ai/generate-image	{ prompt: string; modelId?: number; options?: object; }	`{ requestId: string; status: "pending"	"done"
画像生成結果取得	GET	/api/ai/generate-image/{requestId}	—	{ status: string; imageUrl?: string; }	
プロンプト一覧取得	GET	/api/ai/prompts	?userId=&limit=&offset=	{ prompts: { promptId:number; text:string; createdAt:string }[] }	
類似プロンプト取得	GET	/api/ai/prompts/{promptId}/similar	?limit=	{ similarities: { relatedPromptId:number; score: number }[] }	
モデル一覧取得	GET	/api/ai/models	—	{ models: { modelId:number; name:string; version:string; description:string }[] }	
4.5 ドメイン / 解析 / 生物系 API
機能	メソッド	パス	リクエスト型	レスポンス型	備考
金融価格取得	GET	/api/finance/price	?symbol=string&start=timestamp&end=timestamp	{ prices: { timestamp: string; open: number; high: number; low: number; close: number; volume: number }[] }	
掲示板投稿	POST	/api/forum/posts	{ title: string; body: string; }	{ postId: number; }	
掲示板コメント	POST	/api/forum/posts/{postId}/comments	{ body: string; }	{ commentId: number; }	
仮想座標登録	POST	/api/space/coordinate	{ scale: number; x: number; y: number; z: number; metadata?: object; }	{ coordinateId: number; }	
生物解析結果登録	POST	/api/bio/experiments/{experimentId}/results	{ summary: object; filePath: string; }	{ resultId: number; }	
```
5. 運用 / モニタリング / セキュリティ設計

GitHub Actions による CI → ビルド → テスト → デプロイ

ブルー / グリーンデプロイ、カナリアリリース、ロールバック設計

Prometheus / Grafana によるメトリクス収集・可視化

ログ集約（Loki, ELK, Fluentd など）＋ログ検索・可視化

分散トレーシング (OpenTelemetry / Jaeger / Zipkin 等) 導入

アラート設計（閾値、異常検知、通知 Slack / PagerDuty 等）

サービス間の認証 / 認可設計（内部 API のアクセス制御）

鍵・シークレット管理（Kubernetes Secrets, Vault, KMS 等）

復旧・フェイルオーバー設計、バックアップ／リストア体制

6. 拡張可能性・今後の課題

認証方式拡張（WebAuthn / FIDO2 などの導入）

モデルレジストリ・バージョン管理基盤導入

分析クエリを OLAP 層 (ClickHouse, Apache Druid 等) へオフロード

マルチリージョン展開、シャーディング・水平分割戦略

高可用性構成、障害自動復旧設計

性能テスト・負荷テスト計画


---

