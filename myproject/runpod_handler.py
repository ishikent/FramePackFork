import os
import json
import runpod
import traceback

# 既存のLambdaハンドラとモデル初期化関数をインポート
from myproject.serverless_handler import lambda_handler, initialize_models, models_initialized

print("RunPod Worker Cold Boot")

def runpod_worker(job):
    """
    RunPod Serverless のハンドラ関数。
    入力 'job' を受け取り、既存の lambda_handler を呼び出して結果を返す。
    """
    global models_initialized
    print(f"Received job: {job.get('id', 'N/A')}")

    # --- モデル初期化 (コールドスタート時または未初期化の場合) ---
    # lambda_handler内で初期化されるが、念のためここでも確認・実行
    if not models_initialized:
        try:
            print("Initializing models from RunPod worker...")
            initialize_models()
            print("Models initialized successfully.")
        except Exception as init_err:
            print(f"FATAL: Model initialization failed: {init_err}")
            traceback.print_exc()
            # 初期化失敗は致命的エラーとして返す
            return {"error": f"Model initialization failed: {init_err}"}

    # --- 入力処理 ---
    try:
        job_input = job.get("input")
        if not job_input:
            return {"error": "Missing 'input' in job payload."}

        # lambda_handler が期待する event 形式に変換
        # lambda_handler は body が JSON 文字列であることを期待するため dump する
        event = {
            'body': json.dumps(job_input)
            # 必要に応じて他の event フィールド (headers など) を追加可能
        }
        context = {} # Lambda context は空で良い

        print("Calling lambda_handler...")
        result = lambda_handler(event, context)
        print("lambda_handler returned.")

        # --- 出力処理 ---
        status_code = result.get('statusCode', 500)
        body_str = result.get('body', '{}')
        body = json.loads(body_str) # body は JSON 文字列なのでパース

        if status_code == 200:
            # 成功時は lambda_handler の body の内容 (動画b64を含む辞書) をそのまま返す
            print(f"Job {job.get('id', 'N/A')} completed successfully.")
            return body
        else:
            # 失敗時は body 内の error メッセージを抽出して返す
            error_message = body.get('error', 'Unknown error occurred in lambda_handler.')
            print(f"Job {job.get('id', 'N/A')} failed with status {status_code}: {error_message}")
            return {"error": error_message}

    except Exception as e:
        print(f"Error processing job {job.get('id', 'N/A')}: {e}")
        traceback.print_exc()
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- RunPod Serverless ワーカーを開始 ---
print("Starting RunPod serverless worker...")
runpod.serverless.start({"handler": runpod_worker})
