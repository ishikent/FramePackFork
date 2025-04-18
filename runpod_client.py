import runpod
import os
import json
import base64
import time
import argparse
from pathlib import Path

# --- 設定 ---
# APIキーは環境変数から取得することを推奨
# export RUNPOD_API_KEY="YOUR_API_KEY"
# runpod.api_key = os.getenv("RUNPOD_API_KEY")
# もし環境変数に設定しない場合は、以下のように直接設定（非推奨）
# runpod.api_key = "YOUR_RUNPOD_API_KEY"

# ポーリング間隔（秒）
POLL_INTERVAL = 20

# --- 関数 ---
def encode_image_to_base64(image_path: Path) -> str:
    """画像を読み込みBase64エンコードする"""
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise IOError(f"Error reading or encoding image file {image_path}: {e}")

def run_video_generation(endpoint_id: str, input_data: dict):
    """RunPodエンドポイントを非同期で呼び出し、結果を取得する"""

    if not runpod.api_key:
        raise ValueError("RunPod API Key not set. Set the RUNPOD_API_KEY environment variable.")

    print(f"Using Endpoint ID: {endpoint_id}")
    endpoint = runpod.Endpoint(endpoint_id)

    print("Submitting job...")
    job = None # Initialize job to None before the try block
    try:
        # 非同期でジョブを実行
        job = endpoint.run({"input": input_data})
        # job_id = run_request.id # The job object itself is returned
        # print(f"Job submitted successfully. Job ID: {job.id}") # Access id directly from job object - Removed as 'id' attribute might not exist

        start_time = time.time()
        while True:
            # ステータスを確認
            status = job.status() # Use the job object
            print(f"Current job status: {status} (Elapsed: {time.time() - start_time:.2f}s)")

            if status == "COMPLETED":
                print("Job completed!")
                # 結果を取得 (タイムアウトなし、完了しているので)
                output = job.output() # Use the job object
                return output
            elif status in ["FAILED", "CANCELLED"]:
                print(f"Job failed or was cancelled. Status: {status}")
                # エラーの場合も output() を試みる (エラーメッセージが含まれる可能性があるため)
                try:
                    output = job.output() # Use the job object
                    return {"error": f"Job status: {status}", "details": output}
                except Exception as e:
                    return {"error": f"Job status: {status}, failed to get output details: {e}"}
            elif status == "IN_QUEUE" or status == "IN_PROGRESS":
                # 次のポーリングまで待機
                time.sleep(POLL_INTERVAL)
            else:
                # 想定外のステータス
                print(f"Unexpected job status: {status}")
                return {"error": f"Unexpected job status: {status}"}

    except KeyboardInterrupt: # Catch Ctrl+C
        print("\nKeyboardInterrupt detected. Canceling the job...")
        if job: # Check if a job object exists
            try:
                job.cancel() # Use the job object
                print("Job cancellation request sent.")
                # Give RunPod a moment to process cancellation before checking status again or exiting
                time.sleep(2)
                # Optionally check status one last time
                try:
                    final_status = job.status()
                    print(f"Final job status after cancel request: {final_status}")
                except Exception:
                    print("Could not retrieve final status after cancellation.")
                return {"error": "Job cancelled by user."}
            except Exception as cancel_err:
                print(f"Failed to send cancel request: {cancel_err}")
                return {"error": f"Job cancellation failed: {cancel_err}"}
        else:
            print("No active job to cancel.")
            return {"error": "Cancellation requested but no job was active."}

    except Exception as e:
        print(f"An error occurred during API interaction: {e}")
        # If the error happened before job submission, job might be None
        error_msg = str(e)
        if job:
             # If job exists, try to get more details if it failed due to the error
             try:
                 status = job.status()
                 if status in ["FAILED", "CANCELLED"]:
                     output = job.output()
                     error_msg += f" | Job Status: {status} | Details: {output}"
             except Exception:
                 pass # Ignore errors trying to get status/output after another error
        return {"error": error_msg}

# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RunPod Serverless Video Generation Client")
    parser.add_argument("endpoint_id", help="Your RunPod Serverless Endpoint ID")
    parser.add_argument("image_path", help="Path to the input image file")
    parser.add_argument("-p", "--prompt", required=True, help="Positive prompt for video generation")
    parser.add_argument("-n", "--negative_prompt", default="", help="Negative prompt")
    parser.add_argument("-s", "--seed", type=int, default=12345, help="Seed for generation")
    parser.add_argument("-l", "--length", type=float, default=1.0, help="Total video length in seconds")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=7.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--gpu_mem", type=float, default=6.0, help="GPU memory preservation value")
    parser.add_argument("--api_key", help="RunPod API Key (reads from RUNPOD_API_KEY env var if not provided)")


    args = parser.parse_args()

    # APIキーの設定 (引数 > 環境変数)
    if args.api_key:
        runpod.api_key = args.api_key
    elif os.getenv("RUNPOD_API_KEY"):
         runpod.api_key = os.getenv("RUNPOD_API_KEY")
    else:
        print("Error: RunPod API Key not provided via argument (--api_key) or environment variable (RUNPOD_API_KEY).")
        exit(1)


    try:
        # 画像をBase64エンコード
        image_b64 = encode_image_to_base64(Path(args.image_path))

        # APIに渡す入力データを作成
        input_payload = {
            "input_image_b64": image_b64,
            "prompt": args.prompt,
            "n_prompt": args.negative_prompt,
            "seed": args.seed,
            "total_second_length": args.length,
            "steps": args.steps,
            "cfg": args.cfg, # 注意: 現在のハンドラでは cfg=1.0 以外は未テストかも
            "gpu_memory_preservation": args.gpu_mem
            # 他のオプションパラメータも必要に応じて追加
            # "latent_window_size": 9, # デフォルト値を使う場合
            # "gs": 10.0,
            # "rs": 0.0,
            # "use_teacache": True,
        }

        # API呼び出し実行
        result = run_video_generation(args.endpoint_id, input_payload)

        # 結果表示
        print("\n--- Job Result ---")
        print(json.dumps(result, indent=2))
        print("------------------")

        if isinstance(result, dict) and "video_url" in result:
            print(f"\nVideo successfully generated and uploaded to: {result['video_url']}")
        elif isinstance(result, dict) and "error" in result:
             print(f"\nJob failed: {result['error']}")
             if "details" in result:
                 print(f"Details: {result['details']}")
        else:
            print("\nJob finished with unexpected result format.")


    except FileNotFoundError as e:
        print(f"Error: {e}")
    except IOError as e:
        print(f"Error: {e}")
    except ValueError as e:
         print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
