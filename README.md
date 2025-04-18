<p align="center">
    <img src="https://github.com/user-attachments/assets/2cc030b4-87e1-40a0-b5bf-1b7d6b62820b" width="300">
</p>

# FramePack (Docker版 / RunPod Serverless 対応)

このリポジトリは、["Packing Input Frame Context in Next-Frame Prediction Models for Video Generation"](https://lllyasviel.github.io/frame_pack_gitpage/) の公式実装をベースに、Dockerコンテナ内で実行できるように調整したコードを含んでいます。

FramePackは、初期画像とテキストプロンプトから動画を段階的に生成する、次フレーム（次フレームセクション）予測型のニューラルネットワーク構造です。

このバージョンは、RunPod Serverless プラットフォーム上でAPIエンドポイントとして動作するように設定されています (`myproject/runpod_handler.py`)。Dockerコンテナとしてビルドし、RunPodにデプロイして使用します。

## RunPod Serverless 向け Docker セットアップとデプロイ手順

ここでは、RunPod Serverless で使用するために Docker イメージをビルドし、デプロイする手順を説明します。

**前提条件:**

*   Dockerがインストールされていること: [Dockerを入手](https://docs.docker.com/get-docker/)
*   Docker Hub や GitHub Container Registry などのコンテナリポジトリのアカウント。
*   RunPod アカウント: [RunPod](https://runpod.io)
*   **GPUアクセラレーションを利用する場合 (RunPod側で設定):**
    *   NVIDIA製GPUが搭載されていること。
    *   ホストマシンに対応するNVIDIAドライバがインストールされていること。
    *   NVIDIA Container Toolkitがインストール・設定されていること: [インストールガイド](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    *(注意: RunPod Serverless を利用する場合、GPU環境はRunPod側で提供されるため、ローカルでのGPU設定はビルドやローカルテスト時のみ関係します)*

**手順:**

1.  **ファイルの準備:**
    *   このリポジトリをクローンまたはダウンロードします。
    *   以下のディレクトリ構造になっていることを確認してください:
        ```
        your_project_directory/
        ├── Dockerfile
        ├── requirements.txt
        ├── myproject/
        │   ├── serverless_handler.py # 内部ロジックで使用
        │   └── runpod_handler.py     # RunPod エントリーポイント
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
        *   **モデルの事前ダウンロード:** `Dockerfile` には、ビルド中にモデルをダウンロードするオプションのステップ（ステップ7）が含まれています。これを有効にすると、ビルド時間が大幅に増加し（非常に長くなる可能性があります）、イメージサイズも非常に大きく（数十GB）なりますが、コンテナの初回起動は速くなります。`Dockerfile` でこのステップをコメントアウトすると、モデルは初回実行時にダウンロードされます（RunPodのコールドスタート時間が長くなる可能性があります）。**RunPod Serverless では、通常は事前ダウンロードが推奨されます。**
        *   **Hugging Face トークン (オプション):** モデルが認証を必要とする場合（例: プライベートリポジトリ）、ビルド時に `--build-arg` を使用して Hugging Face トークンを提供する必要があります:
            ```bash
            # Linux/macOS の例 (事前にトークンを設定)
            # export HF_TOKEN="hf_YOUR_TOKEN_HERE" # 実際のトークンに置き換える
            # docker build --build-arg HF_TOKEN=${HF_TOKEN} -t my-video-app .

            # Windows PowerShell の例 (事前にトークンを設定)
            # $env:HF_TOKEN="hf_YOUR_TOKEN_HERE"
            # docker build --build-arg HF_TOKEN=$env:HF_TOKEN -t my-video-app .
            ```

3.  **Dockerイメージのプッシュ:**
    *   ビルドしたイメージをコンテナリポジトリにプッシュします。
        ```bash
        # 例: Docker Hub の場合 (事前に docker login が必要)
        docker tag my-video-app your-dockerhub-username/my-video-app:latest
        docker push your-dockerhub-username/my-video-app:latest
        ```
        *   `your-dockerhub-username/my-video-app:latest` は、ご自身のレジストリとリポジトリ名に合わせて変更してください。

4.  **RunPod Serverless の設定:**
    *   RunPod コンソール ([https://runpod.io/console/serverless/user/templates](https://runpod.io/console/serverless/user/templates)) に移動します。
    *   "New Template" をクリックし、プッシュしたDockerイメージ名 (`your-dockerhub-username/my-video-app:latest` など) を指定します。プライベートリポジトリの場合は、RunPodのユーザー設定で認証情報を追加してください。
    *   必要に応じてコンテナのディスクサイズや環境変数 (例: `HF_TOKEN`) を設定します。
    *   次に、APIエンドポイントを作成します ([https://runpod.io/console/serverless/user/apis](https://runpod.io/console/serverless/user/apis))。
    *   "New API" をクリックし、先ほど作成したテンプレートを選択します。
    *   アイドル時のワーカー数 (Idle Workers) や最大ワーカー数 (Max Workers) などを設定します。**デバッグ目的以外では、アイドルワーカー数を0に設定することを推奨します（課金を避けるため）。**
    *   APIエンドポイントが作成されると、エンドポイントIDが払い出されます。

5.  **APIのテスト:**
    *   `curl` や他のAPIクライアントを使用して、作成したエンドポイントにリクエストを送信します。RunPod API Key が必要です（ユーザー設定で生成）。
        ```bash
        # リクエスト例 (your_endpoint_id と Your_RunPod_API_Key を置き換える)
        curl -X POST https://api.runpod.ai/v2/your_endpoint_id/run \
          -H 'Content-Type: application/json' \
          -H 'Authorization: Bearer Your_RunPod_API_Key' \
          -d '{
            "input": {
              "input_image_b64": "ここに画像のBase64文字列...",
              "prompt": "A beautiful landscape painting",
              "seed": 123,
              "total_second_length": 3.0
            }
          }'
        ```
    *   上記コマンドはジョブIDを返します。ステータス確認エンドポイント (`/status/{job_id}`) を使用して結果を取得します。

## ローカルでのテスト (RunPod Worker)

RunPod ワーカーをローカルでテストするには、`runpod` ライブラリのローカルテスト機能を使用します。

1.  プロジェクトのルートディレクトリに `test_input.json` という名前のファイルを作成します。内容はAPIに送る `"input"` 部分と同じにします。
    ```json
    {
        "input": {
            "input_image_b64": "ここにテスト用の短い画像のBase64文字列...",
            "prompt": "A test prompt",
            "seed": 456,
            "total_second_length": 1.0
        }
    }
    ```
    *(注意: `input_image_b64` には実際のBase64エンコードされた画像データを入れてください。テスト用に小さな画像を使うと良いでしょう。)*

2.  Dockerコンテナをビルドせずに、ローカルのPython環境（必要な依存関係がインストールされている）で直接 `runpod_handler.py` を実行します。
    ```bash
    # プロジェクトルートから実行
    python myproject/runpod_handler.py
    ```
    *   `runpod` ライブラリは `test_input.json` を検出し、それを `job` として `runpod_worker` 関数に渡します。
    *   コンソールに処理ログと最終的な出力（成功時は動画のBase64を含むJSON、失敗時はエラーJSON）が表示されます。

*   **Dockerコンテナ内でのローカルテスト:** Dockerコンテナを実行してテストすることも可能です。`test_input.json` をコンテナ内にコピーするか、ボリュームマウントする必要があります。
    ```bash
    # test_input.json をマウントして実行する例 (Linux/macOS)
    docker run --rm --gpus all -v "$(pwd)/test_input.json:/app/test_input.json" my-video-app
    ```

---

*(以下、元のREADMEにあったアクセス方法などは削除またはコメントアウト)*
<!--
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
-->
