FROM ubuntu:22.04

COPY go1.24.0.linux-amd64.tar.gz /tmp/go.tar.gz
# システム更新と必要パッケージのインストール
RUN apt-get update && apt-get install -y \
    wget \
    time \
    git \
    unzip \
    tree \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ======== Go のインストール ========
ENV GOLANG_VERSION=1.24.0
RUN tar -C /usr/local -xzf /tmp/go.tar.gz && rm /tmp/go.tar.gz


# Go のパス設定
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH=/go
ENV PATH="$GOPATH/bin:$PATH"

# アプリ配置
WORKDIR /app

# go.modを先にコピー（キャッシュ活用）
COPY ./app/go.mod ./app/go.sum* ./
RUN go mod download

# ソースコードをコピー
COPY ./app .

# ビルド
# RUN go build -o server ./main.go

# expose
EXPOSE 8080

# 起動コマンド
# CMD ["./server"]
