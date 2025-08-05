# ベースイメージに軽量なPython環境を使用
FROM python:3.12-slim

# 必要なパッケージをインストール（例: libpq-dev はPostgreSQLクライアントのため）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ設定
WORKDIR /app

# 依存関係を先にコピーしてキャッシュ活用
COPY requirements.txt /app/

# 依存関係をインストール
RUN pip install --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt --resume-retries=5

# アプリケーションコードをコピー
COPY . /app/
# ポートの指定
EXPOSE 8000

# 環境変数でPythonのバッファリング設定（ログがすぐ見えるように）
ENV PYTHONUNBUFFERED=1

# サーバ起動コマンド（開発環境ではrunserverでよいが本番はGunicorn推奨）
CMD ["gunicorn", "search.wsgi:application", "--bind", "0.0.0.0:8000"]
