FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# kind バイナリをインストール
RUN curl -Lo /usr/local/bin/kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64 \
    && chmod +x /usr/local/bin/kind

# kubectl を入れる
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Entrypoint を kind コマンドとしておく（またはシェルを使ってコントロール）
ENTRYPOINT ["kind"]
CMD ["help"]