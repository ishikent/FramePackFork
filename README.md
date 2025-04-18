<p align="center">
    <img src="https://github.com/user-attachments/assets/2cc030b4-87e1-40a0-b5bf-1b7d6b62820b" width="300">
</p>

# FramePack (Docker版)

このリポジトリは、["Packing Input Frame Context in Next-Frame Prediction Models for Video Generation"](https://lllyasviel.github.io/frame_pack_gitpage/) の公式実装をベースに、Dockerコンテナ内で実行できるように調整したコードを含んでいます。

FramePackは、初期画像とテキストプロンプトから動画を段階的に生成する、次フレーム（次フレームセクション）予測型のニューラルネットワーク構造です。

このバージョンは、Dockerコンテナ内でPythonスクリプト (`myproject/serverless_handler.py`) を介して実行するように設定されており、コンテナ化された環境でのテストやデプロイに適しています。

## Dockerのセットアップと実行手順

ここでは、Dockerを使用してアプリケーションをビルドし、実行する手順を説明します。

**前提条件:**

*   Dockerがインストールされていること: [Dockerを入手](https://docs.docker.com/get-docker/)
*   **GPUアクセラレーションを利用する場合:**
    *   NVIDIA製GPUが搭載されていること。
    *   ホストマシンに対応するNVIDIAドライバがインストールされていること。
    *   NVIDIA Container Toolkitがインストール・設定されていること: [インストールガイド](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**手順:**

1.  **ファイルの準備:**
    *   このリポジトリをクローンまたはダウンロードします。
    *   以下のディレクトリ構造になっていることを確認してください:
        ```
        your_project_directory/
        ├── Dockerfile
        ├── requirements.txt
        ├── myproject/
        │   └── serverless_handler.py
        │   └── ... (その他必要なファイル)
        └── diffusers_helper/
            └── __init__.py
            └── ... (その他必要なヘルパーファイル)
        ```

2.  **Dockerイメージのビルド:**
    *   ターミナルまたはコマンドプロンプトを開きます。
    *   プロジェクトのルートディレクトリ（`Dockerfile` がある場所）に移動します。
    *   以下のビルドコマンドを実行します:
        ```bash
        docker build -t my-video-app .
        ```
        *   `-t my-video-app`: 作成するイメージに `my-video-app` という名前（タグ）を付けます。好きな名前に変更可能です。
        *   `.`: ビルドコンテキストとしてカレントディレクトリを指定します。

    *   **ビルドに関する重要な注意点:**
        *   **モデルの事前ダウンロード:** `Dockerfile` には、ビルド中にモデルをダウンロードするオプションのステップ（ステップ7）が含まれています。これを有効にすると、ビルド時間が大幅に増加し（非常に長くなる可能性があります）、イメージサイズも非常に大きく（数十GB）なりますが、コンテナの初回起動は速くなります。`Dockerfile` でこのステップをコメントアウトすると、モデルは初回実行時にダウンロードされます（リソースが限られた環境ではタイムアウトの原因になる可能性があります）。
        *   **Hugging Face トークン (オプション):** モデルが認証を必要とする場合（例: プライベートリポジトリ）、ビルド時に `--build-arg` を使用してHugging Faceトークンを提供する必要があります:
            ```bash
            # Linux/macOS の例 (事前にトークンを設定)
            # export HF_TOKEN="hf_YOUR_TOKEN_HERE"
            # docker build --build-arg HF_TOKEN=${HF_TOKEN} -t my-video-app .

            # Windows PowerShell の例 (事前にトークンを設定)
            # $env:HF_TOKEN="hf_YOUR_TOKEN_HERE"
            # docker build --build-arg HF_TOKEN=$env:HF_TOKEN -t my-video-app .
            ```

3.  **Dockerコンテナの実行:**
    *   イメージのビルドが完了したら、コンテナを実行します。このコマンドは `myproject/serverless_handler.py` で定義されているローカルテストを実行します。
    *   **GPUを使用する場合:**
        ```bash
        docker run --rm --gpus all my-video-app
        ```
    *   **CPUのみを使用する場合:**
        ```bash
        docker run --rm my-video-app
        ```
        *   `--rm`: コンテナ終了時に自動的にコンテナを削除します。
        *   `--gpus all`: ホストの全てのGPUをコンテナから利用可能にします（NVIDIA Container Toolkitが必要です）。

    *   **期待される出力:**
        *   モデルの初期化と動画生成の進捗を示すログがコンソールに出力されます。
        *   成功した場合、成功メッセージが表示され、`local_test_output.mp4` がコンテナ**内部**に保存されたことが示されます。
        *   エラーが発生した場合、エラーメッセージとスタックトレースが表示されます。

4.  **生成された動画ファイルへのアクセス:**
    *   生成された `local_test_output.mp4` は、コンテナ内の `/app/local_test_output.mp4` に作成されます。ホストマシンからアクセスするには、以下のいずれかの方法を使用します。
    *   **方法A: 実行後にコピー (`docker cp` を使用)**
        1.  コンテナIDを調べます（`docker ps -a` を実行して最近のコンテナをリスト表示します）。
        2.  ファイルをコピーします: `docker cp <コンテナID>:/app/local_test_output.mp4 ./` （カレントディレクトリにコピーされます）。
    *   **方法B: 実行時にボリュームをマウント (`-v` を使用)**
        *   まず、ホストマシンに `output` ディレクトリを作成します。
        *   ボリュームマウントを指定してコンテナを実行します:
            ```bash
            # Linux/macOS の例
            docker run --rm --gpus all -v "$(pwd)/output:/app" my-video-app

            # Windows PowerShell の例
            # docker run --rm --gpus all -v "${PWD}/output:/app" my-video-app
            ```
        *   これにより、ホストマシンの `output` フォルダがコンテナ内の `/app` にマウントされ、`local_test_output.mp4` がホストの `output` フォルダ内に直接作成されます。
