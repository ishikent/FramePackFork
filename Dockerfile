# 1. ベースイメージの選択 (PyTorch + CUDA)
# PyTorchとCUDAのバージョンは、使用するモデルやライブラリの要件に合わせて調整してください。
# 例: PyTorch 2.3.0, CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 2. システム依存関係のインストール
# save_bcthw_as_mp4 で ffmpeg が必要になるためインストールします。
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. 環境変数の設定
ENV PYTHONUNBUFFERED=1
# コンテナ内のHugging Faceキャッシュディレクトリを指定
ENV HF_HOME=/app/hf_cache
# ビルド時に --build-arg HF_TOKEN=your_token で渡すための設定
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# 4. 作業ディレクトリの作成と設定
WORKDIR /app
# キャッシュディレクトリを作成
RUN mkdir -p ${HF_HOME}

# 5. requirements.txt のコピーと依存関係のインストール
# requirements.txt を先にコピーして依存関係をインストールすることで、
# コード変更時にも依存関係のレイヤーキャッシュが効くようにします。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. プロジェクトコードのコピー
# myproject と diffusers_helper ディレクトリをコンテナ内の作業ディレクトリにコピー
COPY myproject/ ./myproject/
COPY diffusers_helper/ ./diffusers_helper/

# 7. モデルの事前ダウンロード (オプション、注意点あり)
# このステップを実行すると、Dockerイメージビルド時にモデルがダウンロードされ、
# コンテナ起動後の初回実行が速くなりますが、イメージサイズが非常に大きくなります (数十GB)。
# また、ビルドに非常に時間がかかります。
# モデルを実行時にダウンロードさせたい場合や、ホストからマウントする場合は、
# 以下の RUN コマンドをコメントアウトしてください。
# Hugging Face Hubのプライベートリポジトリ等へのアクセスにトークンが必要な場合は、
# docker build コマンドで --build-arg HF_TOKEN=your_token のように指定してください。
RUN echo "Attempting to pre-download models. This may take a very long time and increase image size significantly." && \
    python -c "from myproject.serverless_handler import initialize_models; initialize_models()" && \
    echo "Model pre-download attempt finished."
# モデル事前ダウンロードは serverless_handler ではなく runpod_handler 経由で行う場合
# (ただし、runpod.serverless.start が実行されてしまうため、通常はコメントアウト推奨)
# RUN echo "Attempting to pre-download models via runpod_handler (usually not recommended here)..." && \
#     python -c "from myproject.serverless_handler import initialize_models; initialize_models()" && \
#     echo "Model pre-download attempt finished."


# 8. 実行コマンド
# コンテナ起動時に myproject/runpod_handler.py を実行します。
# これにより RunPod ワーカーが起動し、APIリクエストを待ち受けます。
# -u オプションは Python の出力をバッファリングしないようにします (RunPod推奨)。
CMD ["python", "-u", "myproject/runpod_handler.py"]
