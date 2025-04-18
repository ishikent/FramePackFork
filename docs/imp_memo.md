# 実装メモ: サーバーレス環境対応 (修正版)

## 目的

`demo_gradio.py` のコアロジックを利用し、サーバーレス環境 (AWS Lambdaなど) でスクリプトとして完結する動画生成機能を実現する。入力はJSON形式、出力もJSON形式（動画データはBase64エンコード）とする。Gradio UIは使用しない。

## 実装方針

1.  **ハンドラ関数の作成 (`serverless_handler.py`)**
    *   サーバーレスプラットフォームのエントリーポイントとなる `lambda_handler(event, context)` 関数を定義する。
    *   このファイルが処理全体の主体となる。

2.  **モデル初期化 (`initialize_models`)**
    *   `demo_gradio.py` のモデルロード・設定ロジックを基に、`initialize_models()` 関数を `serverless_handler.py` 内に定義する。
    *   `high_vram = False` とし、`DynamicSwapInstaller` を適用する。
    *   この関数をハンドラ関数の外（グローバルスコープ）で呼び出し、初期化済みモデルオブジェクトをグローバル変数として保持する（コールドスタート対策）。

3.  **コアロジック関数 (`generate_video_sync`) の作成**
    *   `demo_gradio.py` の `worker` 関数のロジックをベースに、同期的な `generate_video_sync` 関数を `serverless_handler.py` 内または別ファイル (`core_logic.py`) に作成する。
    *   **入力:** JSONからパースされたパラメータ (`input_image` (NumPy), `prompt`, `seed` 等) と、グローバルなモデルオブジェクト群。
    *   **処理:**
        *   `worker` 関数の主要ステップ（テキストエンコード、画像前処理、VAEエンコード、CLIP Visionエンコード、サンプリングループ、VAEデコード）を実行。
        *   `AsyncStream` や `callback` (進捗/プレビュー) 関連コードは削除。ログ出力は可。
        *   メモリ管理 (`load_model_as_complete`, `unload_complete_models` 等) は維持。
        *   サンプリングループ完了後、最終ピクセルデータ (`history_pixels`) を取得。
        *   `save_bcthw_as_mp4` で一時ディレクトリ (`/tmp/`) に動画を保存。
        *   保存したMP4ファイルのパスを返す。
        *   エラーハンドリング (`try...except`) を実装。

4.  **ハンドラ関数 (`lambda_handler`) の実装**
    *   `event` からJSONボディをパースし、パラメータを抽出・検証。
    *   `input_image_b64` をデコードしてNumPy配列 `input_image` に変換。
    *   （必要なら）入力画像を一時保存。
    *   `generate_video_sync` を呼び出す。
    *   返された一時動画ファイルのパスを取得。
    *   動画ファイルを読み込み、Base64エンコード。
    *   一時ファイルを削除。
    *   Base64エンコードされた動画データを含むJSONレスポンスを返す。
    *   エラー発生時はエラー情報を返す。

5.  **`demo_gradio.py` の利用**
    *   `demo_gradio.py` は変更しない。
    *   `serverless_handler.py` は `demo_gradio.py` から必要な関数 (`encode_prompt_conds`, `vae_encode`, `sample_hunyuan` 等)、モデルクラス、メモリ管理関数等をインポートして使用する。`worker` 関数自体は直接呼び出さない。

6.  **依存関係**
    *   `requirements.txt` に必要なライブラリをリストアップする。

## ファイル構成

```
.
├── demo_gradio.py
├── serverless_handler.py  # New (or core_logic.py)
├── diffusers_helper/
│   └── ...
├── docs/
│   ├── imp_memo.md      # Updated
│   └── in_out_spec.md
├── outputs/             # Serverlessでは通常不要だが、ローカルテスト用に残す可能性あり
└── requirements.txt     # New or Updated
```

## 注意点・課題

*   **コールドスタート:** モデルロード時間は依然として課題。
*   **メモリ/実行時間/一時ストレージ制限:** サーバーレスプラットフォームの制限に注意。
*   **エラーハンドリング:** `generate_video_sync` 内およびハンドラでのエラー処理を堅牢にする。
