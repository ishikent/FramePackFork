
import os
import json
import base64
import tempfile
import traceback
import time
import numpy as np
import torch
import einops
from PIL import Image
import io # Image.open用にインポートを追加

# --- demo_gradio.py から必要なインポート ---
# モデルとトークナイザー
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
from diffusers import AutoencoderKLHunyuanVideo
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

# ヘルパー関数
from diffusers_helper.hf_login import login # HFログインが必要な場合
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
# -----------------------------------------

# --- グローバル変数 (モデルと設定) ---
# 環境変数 HF_HOME を設定 (必要に応じて調整)
# Lambda環境では /tmp などを利用するか、デプロイパッケージに含める
hf_home_path = os.path.join(tempfile.gettempdir(), 'hf_download')
os.makedirs(hf_home_path, exist_ok=True)
os.environ['HF_HOME'] = hf_home_path
# ローカルテスト用パス (Lambda環境では上記が優先される想定)
# os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '../hf_download')))


# モデルオブジェクト (初期化後に格納)
text_encoder = None
text_encoder_2 = None
tokenizer = None
tokenizer_2 = None
vae = None
feature_extractor = None
image_encoder = None
transformer = None

# 設定
high_vram = False # サーバーレス環境ではFalseを想定
models_initialized = False
# ------------------------------------

def initialize_models():
    """モデルのロードと初期設定を行う"""
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer, models_initialized

    if models_initialized:
        print("Models already initialized.")
        return

    print("Initializing models...")
    start_time = time.time()

    # HFトークンがあればログイン (環境変数 HF_TOKEN を設定)
    hf_token = os.environ.get('HF_TOKEN', None)
    if hf_token:
        try:
            login(hf_token)
        except Exception as e:
            print(f"HF login failed (non-critical): {e}")


    # モデルのロード (CPUへ)
    # TODO: モデルファイルのパス解決をLambda環境に合わせて調整する必要がある可能性
    # 例: 事前にダウンロードしてデプロイパッケージに含める、EFSを利用するなど
    try:
        print("Loading text_encoder...")
        text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        print("Loading text_encoder_2...")
        text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        print("Loading tokenizer...")
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        print("Loading tokenizer_2...")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        print("Loading vae...")
        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
        print("Loading feature_extractor...")
        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        print("Loading image_encoder...")
        image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
        print("Loading transformer...")
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        # ここで初期化失敗としてマークするか、エラーを再送出するか検討
        models_initialized = False # 失敗したことを示す
        raise # ハンドラで捕捉できるように再送出

    # 評価モード
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    # VAE設定 (low vram)
    vae.enable_slicing()
    vae.enable_tiling()

    # Transformer設定
    transformer.high_quality_fp32_output_for_inference = True
    transformer.to(dtype=torch.bfloat16)

    # 他モデルのdtype設定
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    # 勾配計算不要
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    # メモリ管理 (low vram)
    # DynamicSwapInstaller は CUDA が利用可能な場合にのみ有効
    if torch.cuda.is_available():
        print("CUDA available, installing DynamicSwap...")
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        print("CUDA not available, skipping DynamicSwap installation.")


    models_initialized = True
    end_time = time.time()
    print(f"Models initialized in {end_time - start_time:.2f} seconds.")

@torch.no_grad()
def generate_video_sync(
    # --- モデルオブジェクト (デフォルト値なし) ---
    text_encoder_model,
    text_encoder_2_model,
    tokenizer_inst,
    tokenizer_2_inst,
    vae_model,
    feature_extractor_inst,
    image_encoder_model,
    transformer_model,
    # --- 必須パラメータ (デフォルト値なし) ---
    input_image_np: np.ndarray,
    prompt: str,
    # --- オプションパラメータ (デフォルト値あり) ---
    n_prompt: str = "",
    seed: int = 31337,
    total_second_length: float = 5.0,
    latent_window_size: int = 9,
    steps: int = 25,
    cfg: float = 1.0,
    gs: float = 10.0,
    rs: float = 0.0,
    gpu_memory_preservation: float = 6.0,
    use_teacache: bool = True,
) -> tuple[str, str]: # (output_video_path, input_image_save_path) を返すように変更
    """
    入力パラメータとモデルオブジェクトを受け取り、動画を生成して一時ファイルパスを返す同期関数。
    demo_gradio.py の worker 関数のロジックをベースとする。
    """
    print("Starting video generation...")
    job_id = generate_timestamp()
    output_filename = None
    temp_dir = tempfile.gettempdir()
    # 入力画像保存パスをここで確定
    input_image_save_path = os.path.join(temp_dir, f'{job_id}_input.png')

    # CUDAが利用可能かチェック
    cuda_available = torch.cuda.is_available()
    target_device = gpu if cuda_available else cpu
    print(f"Target device: {target_device}")

    try:
        # --- worker 関数のロジック開始 ---
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        print(f"Total latent sections: {total_latent_sections}")

        # GPUクリーン (low vram かつ CUDA利用可能時)
        if not high_vram and cuda_available:
            unload_complete_models(
                text_encoder_model, text_encoder_2_model, image_encoder_model, vae_model, transformer_model
            )

        # テキストエンコード
        print("Text encoding...")
        if cuda_available:
            fake_diffusers_current_device(text_encoder_model, target_device)
            load_model_as_complete(text_encoder_2_model, target_device=target_device)
        else:
            # CPUの場合は明示的にCPUへ
            text_encoder_model.to(cpu)
            text_encoder_2_model.to(cpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder_model, text_encoder_2_model, tokenizer_inst, tokenizer_2_inst)
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder_model, text_encoder_2_model, tokenizer_inst, tokenizer_2_inst)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # 画像前処理
        print("Image processing...")
        H, W, C = input_image_np.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_resized_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)
        # 一時保存 (デバッグや確認用)
        try:
            Image.fromarray(input_image_resized_np).save(input_image_save_path)
            print(f"Input image saved to {input_image_save_path}")
        except Exception as img_save_err:
            print(f"Warning: Could not save input image to {input_image_save_path}: {img_save_err}")

        input_image_pt = torch.from_numpy(input_image_resized_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAEエンコード
        print("VAE encoding...")
        if cuda_available:
            load_model_as_complete(vae_model, target_device=target_device)
        else:
            vae_model.to(cpu)
        start_latent = vae_encode(input_image_pt, vae_model)

        # CLIP Visionエンコード
        print("CLIP Vision encoding...")
        if cuda_available:
            load_model_as_complete(image_encoder_model, target_device=target_device)
        else:
            image_encoder_model.to(cpu)
        image_encoder_output = hf_clip_vision_encode(input_image_resized_np, feature_extractor_inst, image_encoder_model)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype変換
        llama_vec = llama_vec.to(transformer_model.dtype)
        llama_vec_n = llama_vec_n.to(transformer_model.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer_model.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer_model.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer_model.dtype)

        # サンプリング準備
        print("Preparing for sampling...")
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3
        # history_latents は float32 で CPU に保持
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
             latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:
             latent_paddings = list(reversed(range(total_latent_sections))) # 元のロジックも残す

        # サンプリングループ
        for i, latent_padding in enumerate(latent_paddings):
            section_start_time = time.time()
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size
            print(f"Sampling section {i+1}/{total_latent_sections} (padding={latent_padding}, last={is_last_section})...")

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # clean_latents* は history_latents (CPU, float32) から取得
            clean_latents_pre = start_latent.to(history_latents.device, dtype=history_latents.dtype) # start_latentも合わせる
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # Transformerをターゲットデバイスへ
            if cuda_available:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer_model, target_device=target_device, preserved_memory_gb=gpu_memory_preservation)
            else:
                transformer_model.to(cpu)


            # TeaCache設定
            if use_teacache:
                transformer_model.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer_model.initialize_teacache(enable_teacache=False)

            # sample_hunyuan 呼び出し (callbackなし)
            # 必要なテンソルをターゲットデバイスと適切なdtypeに移動
            generated_latents = sample_hunyuan(
                transformer=transformer_model,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd, # GeneratorはCPUのまま渡す
                prompt_embeds=llama_vec.to(target_device),
                prompt_embeds_mask=llama_attention_mask.to(target_device),
                prompt_poolers=clip_l_pooler.to(target_device),
                negative_prompt_embeds=llama_vec_n.to(target_device),
                negative_prompt_embeds_mask=llama_attention_mask_n.to(target_device),
                negative_prompt_poolers=clip_l_pooler_n.to(target_device),
                device=target_device, # サンプラー内部で使用するデバイス
                dtype=torch.bfloat16 if cuda_available else torch.float32, # CPUではbfloat16非対応の場合あり
                image_embeddings=image_encoder_last_hidden_state.to(target_device),
                latent_indices=latent_indices.to(target_device),
                clean_latents=clean_latents.to(target_device, dtype=transformer_model.dtype),
                clean_latent_indices=clean_latent_indices.to(target_device),
                clean_latents_2x=clean_latents_2x.to(target_device, dtype=transformer_model.dtype),
                clean_latent_2x_indices=clean_latent_2x_indices.to(target_device),
                clean_latents_4x=clean_latents_4x.to(target_device, dtype=transformer_model.dtype),
                clean_latent_4x_indices=clean_latent_4x_indices.to(target_device),
                callback=None, # サーバーレスではコールバック不要
            )

            # 生成されたlatentをCPU, float32に戻す
            generated_latents = generated_latents.to(cpu, dtype=torch.float32)

            if is_last_section:
                # start_latentもCPU, float32に合わせる
                generated_latents = torch.cat([start_latent.to(cpu, dtype=torch.float32), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            # history_latents (CPU, float32) を更新
            history_latents = torch.cat([generated_latents, history_latents], dim=2)

            # VAEデコード
            print("VAE decoding section...")
            if cuda_available:
                offload_model_from_device_for_memory_preservation(transformer_model, target_device=target_device, preserved_memory_gb=8)
                load_model_as_complete(vae_model, target_device=target_device)
            else:
                transformer_model.to(cpu) # TransformerをCPUに戻す
                vae_model.to(cpu) # VAEをCPUに

            # デコード対象のlatent (CPU, float32)
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            # vae_decodeに渡す前にターゲットデバイスとVAEのdtypeに変換
            real_history_latents_for_decode = real_history_latents.to(target_device, dtype=vae_model.dtype)

            if history_pixels is None:
                # デコード結果をCPUに戻す
                history_pixels = vae_decode(real_history_latents_for_decode, vae_model).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                # デコード対象のlatent (CPU, float32)
                current_latents_for_decode = real_history_latents[:, :, :section_latent_frames].to(target_device, dtype=vae_model.dtype)
                # デコード結果をCPUに戻す
                current_pixels = vae_decode(current_latents_for_decode, vae_model).cpu()
                # soft_append_bcthw はCPU上で実行
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if cuda_available:
                unload_complete_models()

            section_end_time = time.time()
            print(f"Section {i+1} finished in {section_end_time - section_start_time:.2f} seconds. Decoded pixels shape: {history_pixels.shape}")

            if is_last_section:
                break
        # --- ループ終了 ---

        # 最終的な動画を保存
        output_filename = os.path.join(temp_dir, f'{job_id}_output.mp4')
        print(f"Saving final video to {output_filename}...")
        # save_bcthw_as_mp4 はCPUテンソルを受け取る
        save_bcthw_as_mp4(history_pixels, output_filename, fps=30)
        print("Video generation complete.")

    except Exception as e:
        print(f"Error during video generation: {e}")
        traceback.print_exc()
        # エラー時もクリーンアップを試みる (low vram かつ CUDA利用可能時)
        if not high_vram and cuda_available:
            unload_complete_models(
                text_encoder_model, text_encoder_2_model, image_encoder_model, vae_model, transformer_model
            )
        raise # エラーを再送出してハンドラで捕捉

    finally:
        # GPUメモリ解放 (CUDA利用可能時)
        if cuda_available:
             if not high_vram:
                 unload_complete_models(
                     text_encoder_model, text_encoder_2_model, image_encoder_model, vae_model, transformer_model
                 )
             torch.cuda.empty_cache()

    # 保存した入力画像のパスも返す
    return output_filename, input_image_save_path


def lambda_handler(event, context):
    """AWS Lambda ハンドラ関数"""
    global models_initialized
    handler_start_time = time.time()

    # モデル初期化 (コールドスタート時のみ実行、失敗時はエラー)
    if not models_initialized:
        try:
            initialize_models()
        except Exception as init_err:
            print(f"FATAL: Model initialization failed: {init_err}")
            traceback.print_exc()
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': f'Model initialization failed: {init_err}'})
            }


    output_video_path = None
    input_image_temp_path = None

    try:
        # --- 入力パース ---
        print("Parsing input event...")
        # API Gateway v1, v2, Lambda Function URL, direct invokeに対応
        if isinstance(event, dict):
            if 'body' in event:
                body_str = event['body']
                if event.get('isBase64Encoded', False):
                     body_str = base64.b64decode(body_str).decode('utf-8')
                body = json.loads(body_str)
            elif 'prompt' in event and 'input_image_b64' in event: # Direct invoke?
                body = event
            else: # Try to parse event itself as JSON body
                 body = event
        else:
             raise ValueError("Invalid event format")


        # 必須パラメータ
        input_image_b64 = body.get('input_image_b64')
        prompt = body.get('prompt')

        if not input_image_b64 or not prompt:
            raise ValueError("Missing required parameters: 'input_image_b64' and 'prompt'")

        # オプションパラメータ (デフォルト値は generate_video_sync 側で設定)
        n_prompt = body.get('n_prompt', "")
        seed = int(body.get('seed', 31337))
        total_second_length = float(body.get('total_second_length', 5.0))
        latent_window_size = int(body.get('latent_window_size', 9)) # 基本的に変更しない
        steps = int(body.get('steps', 25))
        cfg = float(body.get('cfg', 1.0)) # 基本的に変更しない
        gs = float(body.get('gs', 10.0))
        rs = float(body.get('rs', 0.0)) # 基本的に変更しない
        # gpu_memory_preservation を body から取得、なければデフォルト値 6.0 を使用
        gpu_memory_preservation = float(body.get('gpu_memory_preservation', 6.0))
        use_teacache = bool(body.get('use_teacache', True))

        # 画像デコード
        print("Decoding input image...")
        image_data = base64.b64decode(input_image_b64)
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_image_np = np.array(input_image)
        print(f"Input image decoded, shape: {input_image_np.shape}")
        # -----------------

        # --- コアロジック実行 ---
        generation_start_time = time.time()
        # generate_video_sync から両方のパスを受け取る
        output_video_path, input_image_temp_path = generate_video_sync(
            input_image_np=input_image_np,
            prompt=prompt,
            n_prompt=n_prompt,
            seed=seed,
            total_second_length=total_second_length,
            latent_window_size=latent_window_size,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            # パースした値を渡す
            gpu_memory_preservation=gpu_memory_preservation,
            use_teacache=use_teacache,
            # グローバルなモデルオブジェクトを渡す
            text_encoder_model=text_encoder,
            text_encoder_2_model=text_encoder_2,
            tokenizer_inst=tokenizer,
            tokenizer_2_inst=tokenizer_2,
            vae_model=vae,
            feature_extractor_inst=feature_extractor,
            image_encoder_model=image_encoder,
            transformer_model=transformer
        )
        generation_end_time = time.time()
        print(f"Core generation finished in {generation_end_time - generation_start_time:.2f} seconds.")
        # ---------------------

        # --- 出力処理 ---
        if output_video_path and os.path.exists(output_video_path):
            print(f"Encoding output video {output_video_path} to Base64...")
            with open(output_video_path, "rb") as vf:
                video_data_b64 = base64.b64encode(vf.read()).decode('utf-8')

            # 一時ファイル削除 (finallyブロックに移動)
            # print("Scheduling temporary files for cleanup...")

            handler_end_time = time.time()
            print(f"Handler execution finished in {handler_end_time - handler_start_time:.2f} seconds.")

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'message': 'Video generated successfully.',
                    'output_video_b64': video_data_b64
                })
            }
        else:
            raise RuntimeError("Video generation failed, output file not found or path is invalid.")
        # -------------

    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        traceback.print_exc()
        handler_end_time = time.time()
        print(f"Handler execution failed after {handler_end_time - handler_start_time:.2f} seconds.")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }
    finally:
        # --- 一時ファイルクリーンアップ ---
        print("Cleaning up temporary files...")
        if output_video_path and os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
                print(f"Removed temporary video file: {output_video_path}")
            except OSError as rm_err:
                print(f"Warning: Could not remove temporary video file {output_video_path}: {rm_err}")
        if input_image_temp_path and os.path.exists(input_image_temp_path):
             try:
                 os.remove(input_image_temp_path)
                 print(f"Removed temporary input image file: {input_image_temp_path}")
             except OSError as rm_err:
                 print(f"Warning: Could not remove temporary input image file {input_image_temp_path}: {rm_err}")
        # -----------------------------


# --- グローバルスコープでのモデル初期化呼び出し ---
# Lambda環境では、ハンドラ外のコードはコンテナ初期化時に実行される
# ただし、初期化失敗はハンドラ内で処理する
# initialize_models() # ハンドラ内で呼び出すように変更

# --- ローカルテスト用 ---
if __name__ == '__main__':
    print("Running local test...")
    # ダミーの入力データを作成
    test_image = Image.new('RGB', (640, 360), color = 'blue') # サイズを調整
    buffered = io.BytesIO()
    test_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    test_event = {
        'body': json.dumps({
            'input_image_b64': img_str,
            'prompt': 'A blue rectangle moving',
            'seed': 12345,
            'total_second_length': 1.5 # テスト用にさらに短く
        })
    }
    test_context = {}

    # ハンドラ実行
    result = lambda_handler(test_event, test_context)

    # 結果表示
    print("\n--- Lambda Handler Result ---")
    print(f"Status Code: {result['statusCode']}")
    body_content = json.loads(result['body'])
    if result['statusCode'] == 200:
        print(f"Message: {body_content['message']}")
        print(f"Output video b64 length: {len(body_content['output_video_b64'])}")
        # 必要ならファイルに書き出す
        try:
            with open("local_test_output.mp4", "wb") as f:
                f.write(base64.b64decode(body_content['output_video_b64']))
            print("Output video saved to local_test_output.mp4")
        except Exception as save_err:
            print(f"Could not save output video locally: {save_err}")
    else:
        print(f"Error: {body_content}")
    print("---------------------------\n")
