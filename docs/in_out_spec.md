# FramePack 入出力仕様

このドキュメントは、FramePackアプリケーションの主要な入力と出力を定義します。

## 入力 (Input)

### 1. Gradioインターフェース経由のユーザー指定

*   **入力画像 (`input_image`):**
    *   形式: NumPy配列
    *   説明: 動画生成の元となる静止画像。
*   **プロンプト (`prompt`):**
    *   形式: 文字列
    *   説明: 生成したい動画の内容を表すテキスト。
*   **シード値 (`seed`):**
    *   形式: 整数
    *   説明: 乱数生成の初期値。再現性の確保に使用。
*   **動画の長さ (`total_second_length`):**
    *   形式: 浮動小数点数 (秒)
    *   説明: 生成する動画のおおよその秒数。
*   **ステップ数 (`steps`):**
    *   形式: 整数
    *   説明: 画像生成プロセスの反復回数。
*   **Distilled CFGスケール (`gs`):**
    *   形式: 浮動小数点数
    *   説明: テキストプロンプトへのガイダンスの強さを調整するパラメータ。
*   **GPUメモリ確保量 (`gpu_memory_preservation`):**
    *   形式: 浮動小数点数 (GB)
    *   説明: 推論中に確保しておくGPUメモリ量。OOM (Out Of Memory) エラー対策。
*   **TeaCacheの使用 (`use_teacache`):**
    *   形式: 真偽値
    *   説明: 速度向上のためのキャッシュ機能（TeaCache）を使用するかどうか。
*   **(非表示/内部パラメータ)**
    *   ネガティブプロンプト (`n_prompt`): 文字列
    *   Latent Window Size (`latent_window_size`): 整数
    *   CFG Scale (`cfg`): 浮動小数点数
    *   CFG Re-Scale (`rs`): 浮動小数点数

### 2. コマンドライン引数 (demo_gradio.py 実行時)

*   `--share`: Gradioの共有リンクを生成するかどうか。
*   `--server`: GradioサーバーがリッスンするIPアドレス (デフォルト: '0.0.0.0')。
*   `--port`: Gradioサーバーが使用するポート番号。
*   `--inbrowser`: 起動時にブラウザを自動で開くかどうか。

### 3. 環境変数

*   `HF_HOME`: Hugging Face関連ファイル（モデル、キャッシュ等）のダウンロード先ディレクトリパス。
*   `HF_TOKEN` (オプション): Hugging Face Hubへのログイン認証トークン。プライベートモデル等へのアクセスに必要。

### 4. 事前学習済みモデルファイル

*   テキストエンコーダー (Llama, CLIP)
*   トークナイザー (Llama, CLIP)
*   VAE (Variational Autoencoder)
*   画像エンコーダー (Siglip Vision)
*   画像特徴抽出器 (Siglip Image Processor)
*   Transformerモデル (HunyuanVideoTransformer3DModelPacked)
    *   注意: これらのモデルは、指定されたリポジトリ (例: "hunyuanvideo-community/HunyuanVideo", "lllyasviel/flux_redux_bfl", "lllyasviel/FramePackI2V_HY") から自動的にダウンロード・キャッシュされます。初回実行時にはダウンロード時間が必要です。

## 出力 (Output)

### 1. Gradioインターフェースへの表示

*   **生成された動画 (`result_video`):**
    *   形式: MP4動画
    *   説明: 最終的に生成された動画。自動再生・ループ表示される。
*   **プレビュー画像 (`preview_image`):**
    *   形式: 画像 (PNG等)
    *   説明: 生成途中の潜在空間の状態を簡易的にデコードした静止画像。リアルタイムで更新される。
*   **進捗テキスト (`progress_desc`):**
    *   形式: 文字列
    *   説明: 現在の処理ステップ、生成済みフレーム数、推定経過時間などのテキスト情報。
*   **プログレスバー (`progress_bar`):**
    *   形式: HTML (Gradioコンポーネント)
    *   説明: 処理全体の進捗状況を視覚的に示すバー。

### 2. ファイルシステムへの保存 (`./outputs/` ディレクトリ)

*   **入力画像の保存:**
    *   ファイル名形式: `{job_id}.png`
    *   説明: ユーザーがアップロードし、処理用にリサイズ・クロップされた入力画像。`{job_id}` は実行ごとに一意なタイムスタンプベースのID。
*   **生成動画の保存:**
    *   ファイル名形式: `{job_id}_{total_latent_frames}.mp4`
    *   説明: `worker` 関数内の動画伸長ループの各イテレーションで、その時点までの生成結果がMP4ファイルとして保存される。ファイル名の `{total_latent_frames}` はその時点での潜在空間のフレーム数。最終的に生成されるファイルが最も長い動画となる。

### 3. コンソール (標準出力)

*   起動時の設定情報 (検出されたVRAM容量、コマンドライン引数など)。
*   モデルのロード状況やキャッシュ利用に関するログ。
*   メモリ管理 (モデルのオフロード/ロード) に関するログ。
*   `worker` 関数内の主要な処理ステップに関するデバッグ情報 (例: テキストエンコード開始、VAEエンコード開始、サンプリング開始、各セクションのパディングサイズなど)。
*   エラー発生時のPythonトレースバック情報。
