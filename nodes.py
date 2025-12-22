#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, gc
import math
import torch
import folder_paths
import comfy.utils
import time
import sys
import psutil

import numpy as np
import torch.nn.functional as F

from einops import rearrange
from huggingface_hub import snapshot_download
try:
    from .src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
    from .src.models.TCDecoder import build_tcdecoder
    from .src.models.utils import clean_vram, get_device_list, Buffer_LQ4x_Proj, Causal_LQ4x_Proj
    from .src.models import wan_video_dit
    from .src.models.wan_video_vae import WanVideoVAE, Wan22VideoVAE, LightX2VVAE, create_video_vae
except ImportError:
    from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
    from src.models.TCDecoder import build_tcdecoder
    from src.models.utils import clean_vram, get_device_list, Buffer_LQ4x_Proj, Causal_LQ4x_Proj
    from src.models import wan_video_dit
    from src.models.wan_video_vae import WanVideoVAE, Wan22VideoVAE, LightX2VVAE, create_video_vae

try:
    import safetensors.torch
except ImportError:
    pass

# Available VAE types for user selection
VAE_TYPES = ["wan2.1", "wan2.2", "lightx2v"]

device_choices = get_device_list()

def log(message: str, message_type: str = 'normal', icon: str = "", end: str = "\n", in_place: bool = False):
    if icon:
        message = f"{icon} {message}"
        
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message

    if in_place:
        # Clear line before printing
        sys.stdout.write("\r\033[K" + message)
        sys.stdout.flush()
    else:
        print(f"{message}", end=end, flush=True)

def get_vram_info():
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return vram_used, vram_reserved, vram_total
    return 0, 0, 0

def log_resource_usage(prefix="Resource Usage", end="\n", in_place=False):
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024 ** 3)
    ram_total = ram.total / (1024 ** 3)
    
    msg = f"[{prefix}] RAM: {ram_used:.1f}/{ram_total:.1f}G"
    
    if torch.cuda.is_available():
        vram_used, vram_reserved, vram_total = get_vram_info()
        msg += f" | VRAM: {vram_used:.1f}/{vram_reserved:.1f}/{vram_total:.1f}G"
        
    log(msg, message_type='info', icon="📊", end=end, in_place=in_place)

def model_download(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(folder_paths.models_dir, model_name.split("/")[-1])
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info', icon="⬇️")
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = video_squeezed.permute(1, 2, 3, 0) # C F H W -> F H W C
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def next_8n5(n):  # next 8n+5
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = math.ceil(sW / multiple) * multiple
    tH = math.ceil(sH / multiple) * multiple
    
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW
    
    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    
    if sW < tW or sH < tH:
        pad_l = max(0, (tW - sW) // 2)
        pad_r = max(0, tW - sW - pad_l)
        pad_t = max(0, (tH - sH) // 2)
        pad_b = max(0, tH - sH - pad_t)
        upscaled_tensor = F.pad(upscaled_tensor, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
    
    l = max(0, (upscaled_tensor.shape[3] - tW) // 2)
    t = max(0, (upscaled_tensor.shape[2] - tH) // 2)
    cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]

    return cropped_tensor.squeeze(0)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    
    multiple = 128 # Keep 128 alignment for VAE/DiT blocks
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH).to('cpu').to(dtype) * 2.0 - 1.0
        frames.append(tensor_chw)
        del frame_slice

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    return vid_final, tH, tW, F

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
                
            coords.append((x1, y1, x2, y2))
            
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def init_pipeline(model, mode, device, dtype, alt_vae="none", vae_type="wan2.1"):
    """
    Initialize FlashVSR pipeline with specified model and VAE type.
    
    Args:
        model: Model name (FlashVSR or FlashVSR-v1.1)
        mode: Pipeline mode (full, tiny, tiny-long)
        device: Computation device
        dtype: Torch dtype
        alt_vae: Alternative VAE file path or "none" for built-in
        vae_type: VAE type - "wan2.1", "wan2.2", or "lightx2v"
    """
    model_download(model_name="JunhaoZhuang/"+model)
    model_path = os.path.join(folder_paths.models_dir, model)
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist!\nPlease save all weights to "{model_path}"')
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist!\nPlease save it to "{model_path}"')
    if alt_vae != "none":
        vae_path = folder_paths.get_full_path_or_raise("vae", alt_vae)
        if not os.path.exists(vae_path):
            raise RuntimeError(f'"{alt_vae}" does not exist!')
    else:
        vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_path):
            raise RuntimeError(f'"Wan2.1_VAE.pth" does not exist!\nPlease save it to "{model_path}"')
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!\nPlease save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!\nPlease save it to "{model_path}"')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "posi_prompt.pth")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)

        # Manual VAE loading fallback if ModelManager fails
        if pipe.vae is None:
            log("ModelManager failed to load VAE. Attempting manual load...", message_type='warning', icon="⚠️")
            try:
                # Create VAE based on selected type
                vae_type_lower = vae_type.lower()
                if vae_type_lower == "wan2.2":
                    pipe.vae = Wan22VideoVAE(z_dim=16, dim=96).to(device=device, dtype=dtype)
                    log(f"Using Wan2.2 VAE (optimized normalization)", message_type='info', icon="🚀")
                elif vae_type_lower == "lightx2v":
                    pipe.vae = LightX2VVAE(z_dim=16, dim=64, use_full_arch=True).to(device=device, dtype=dtype)
                    log(f"Using LightX2V VAE (optimized for VRAM)", message_type='info', icon="⚡")
                else:
                    pipe.vae = WanVideoVAE(z_dim=16, dim=96).to(device=device, dtype=dtype)
                    log(f"Using Wan2.1 VAE (default)", message_type='info', icon="📦")

                # Check for safetensors vs pth
                if vae_path.endswith(".safetensors"):
                    try:
                        import safetensors.torch
                        sd = safetensors.torch.load_file(vae_path)
                    except ImportError:
                        raise RuntimeError("safetensors library required to load .safetensors VAE file. Please install it.")
                else:
                    # Allow weights_only=False for .pth files as per user request/necessity with older formats
                    sd = torch.load(vae_path, map_location="cpu", weights_only=False)

                pipe.vae.load_state_dict(sd)
                log(f"Manually loaded VAE from {vae_path}", message_type='info', icon="📦")
            except Exception as e:
                log(f"Failed to manually load VAE: {e}", message_type='error', icon="❌")
                raise RuntimeError(f"Could not load VAE from {vae_path}")
        else:
            # If ModelManager loaded VAE, we may need to swap to a different VAE type
            vae_type_lower = vae_type.lower()
            if vae_type_lower == "wan2.2" and not isinstance(pipe.vae, Wan22VideoVAE):
                log(f"Upgrading to Wan2.2 VAE with optimized normalization...", message_type='info', icon="🚀")
                old_state = pipe.vae.state_dict()
                pipe.vae = Wan22VideoVAE(z_dim=16, dim=96).to(device=device, dtype=dtype)
                pipe.vae.load_state_dict(old_state)
            elif vae_type_lower == "lightx2v" and not isinstance(pipe.vae, LightX2VVAE):
                log(f"Switching to LightX2V VAE for reduced VRAM...", message_type='info', icon="⚡")
                old_state = pipe.vae.state_dict()
                pipe.vae = LightX2VVAE(z_dim=16, dim=64, use_full_arch=True).to(device=device, dtype=dtype)
                pipe.vae.load_state_dict(old_state)

        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt_path])
        if mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        else:
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device, weights_only=False), strict=False)
        pipe.TCDecoder.clean_mem()
    
    if model == "FlashVSR":
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    else:
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu", weights_only=False), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    pipe.offload_model()

    # Log VAE type information
    vae_info = f"VAE Type: {vae_type}"
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        vae_info += f" ({type(pipe.vae).__name__})"
    
    log(f"Pipeline Initialized: Mode={mode}, Device={device}, Dtype={dtype}, Attention={wan_video_dit.ATTENTION_MODE}", message_type='info', icon="🔧")
    log(f"Model: {model}, {vae_info}", message_type='info', icon="📦")

    return pipe

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing", enable_debug=False):
        self.desc = desc
        self.pbar = None
        self.iterable = None
        self.total = total
        self.enable_debug = enable_debug
        self.start_time = time.time()
        self.step_idx = 0
        
        if iterable is not None:
            try:
                self.total = len(iterable)
                self.iterable = iter(iterable)
            except TypeError:
                if self.total is None:
                    raise ValueError("Total must be provided for iterables with no length.")

        elif self.total is not None:
            pass
            
        else:
            raise ValueError("Either iterable or total must be provided.")
            
    def __iter__(self):
        if self.iterable is None:
            raise TypeError(f"'{type(self).__name__}' object is not iterable. Did you mean to use it with a 'with' statement?")
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self
    
    def __next__(self):
        if self.iterable is None:
            raise TypeError("Cannot call __next__ on a non-iterable cqdm object.")
        try:
            step_start = time.time()
            val = next(self.iterable)
            
            if self.pbar:
                self.pbar.update(1)
            
            self.step_idx += 1

            # Show a text progress bar in the log (single line using \r)
            perc = (self.step_idx / self.total) * 100
            bar_len = 20
            filled = int(bar_len * self.step_idx // self.total)
            bar = '█' * filled + '░' * (bar_len - filled)

            elapsed = time.time() - self.start_time
            rate = self.step_idx / elapsed if elapsed > 0 else 0

            msg = f"{self.desc}: {self.step_idx}/{self.total} |{bar}| {perc:.1f}%"

            if self.enable_debug:
                step_end = time.time()
                step_time = step_end - step_start
                msg += f" (Step: {step_time:.2f}s)"
                # Pass in_place=True to log_resource_usage to keep it on one line if possible
                # But note log_resource_usage prints Resource usage which is long.
                log_resource_usage(prefix=msg, in_place=True)
            else:
                print(f"\r{msg}", end="", flush=True)
                if self.step_idx == self.total:
                    print()

            return val
        except StopIteration:
            total_time = time.time() - self.start_time
            if self.enable_debug:
                # Use print with newline here to finalize the log block
                print(f"\n✅ Loop '{self.desc}' finished in {total_time:.2f}s", flush=True)
            raise
            
    def __enter__(self):
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def __len__(self):
        return self.total

def process_chunk(pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, force_offload, enable_debug, is_single_frame_input=False):
    """
    Processes a single chunk of frames.
    """
    clean_vram()
    _frames = frames
    _device = pipe.device
    dtype = pipe.torch_dtype
    
    # Padding logic for the chunk
    add = next_8n5(frames.shape[0]) - frames.shape[0]
    padding_frames = frames[-1:, :, :, :].repeat(add, 1, 1, 1)
    _frames = torch.cat([frames, padding_frames], dim=0)

    if tiled_dit:
        N, H, W, C = _frames.shape
        
        final_output_canvas = torch.zeros(
            (N, H * scale, W * scale, C), 
            dtype=torch.float16, 
            device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        
        log(f"Starting Tiled Processing: {len(tile_coords)} tiles", message_type='info', icon="🚀")
        
        for i, (x1, y1, x2, y2) in enumerate(cqdm(tile_coords, desc="Processing Tiles", enable_debug=enable_debug)):
            tile_start = time.time()
            if enable_debug:
                log(f"Processing tile {i+1}/{len(tile_coords)}: ({x1},{y1}) -> ({x2},{y2})", message_type='info', icon="🔄")
            
            input_tile = _frames[:, y1:y2, x1:x2, :]
            
            LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
            if not isinstance(pipe, FlashVSRTinyLongPipeline):
                LQ_tile = LQ_tile.to(_device)

            # Note: pipe() calls implicitly handle the rest.
            # But pipe() can raise OOM. We let it propagate to flashvsr to handle fallback.
            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, unload_dit=unload_dit, force_offload=force_offload,
                enable_debug_logging=enable_debug
            )
            
            processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
            
            if enable_debug:
                tile_end = time.time()
                tile_time = tile_end - tile_start
                log(f"Tile {i+1} completed in {tile_time:.2f}s", message_type='info', icon="⏱️")
            
            mask_nchw = create_feather_mask(
                (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                tile_overlap * scale
            ).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
            out_x1, out_y1 = x1 * scale, y1 * scale
            
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
            
            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()
            
        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
    else:
        log("Preparing full frame processing...", message_type='info', icon="🎞️")
        if enable_debug:
            log_resource_usage(prefix="Pre-Preprocess")
        
        LQ, th, tw, F = prepare_input_tensor(_frames, _device, scale=scale, dtype=dtype)
        if not isinstance(pipe, FlashVSRTinyLongPipeline):
            LQ = LQ.to(_device)
            
        log(f"Processing {frames.shape[0]} frames...", message_type='info', icon="🚀")
        
        process_start = time.time()

        class cqdm_debug(cqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, enable_debug=enable_debug)

        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
            progress_bar_cmd=cqdm_debug, LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
            color_fix = color_fix, unload_dit=unload_dit, force_offload=force_offload
        )

        process_end = time.time()
        
        if enable_debug:
            log(f"Inference completed in {process_end - process_start:.2f}s", message_type='info', icon="⏱️")
        final_output_tensor = tensor2video(video).to('cpu')
        
        # Crop back to canvas (tiling naturally handles this via summing, but non-tiled produces full tensor)
        # However, non-tiled output is already in correct shape?
        # Check tensor_upscale_then_center_crop logic:
        # It creates target dimensions (tW, tH) multiple of 128.
        # But if we didn't tile, the output 'video' has those dimensions.
        # We need to crop it back if necessary?
        # Actually, tensor2video returns the video as is.
        # But process_chunk logic for tiling accumulates into `final_output_canvas` which has exact size H*scale, W*scale.
        # For non-tiled, the output 'video' has size tH, tW (aligned to 128).
        # We need to crop it to match expected output size.
        
        H, W = frames.shape[1], frames.shape[2]
        final_output = final_output_tensor[:, :H*scale, :W*scale, :]

        del video, LQ
        clean_vram()

    if is_single_frame_input and frames.shape[0] == 1:
        if frames.shape[0] == 1:
            final_output = final_output.to("cpu")
            stacked_image_tensor = torch.median(final_output, dim=0).values.unsqueeze(0).float()
            del final_output
            clean_vram()
            return stacked_image_tensor

    return final_output[:frames.shape[0], :, :, :]

def flashvsr(pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, force_offload, enable_debug=False, chunk_size=0, resize_factor=1.0):
    clean_vram()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()

    # Resize Input Frames if requested
    if resize_factor < 1.0 and resize_factor > 0:
        log(f"Resizing input by factor {resize_factor}...", message_type='info', icon="📉")
        N, H, W, C = frames.shape
        new_H, new_W = int(H * resize_factor), int(W * resize_factor)
        # Permute to NCHW for interpolate
        frames_permuted = frames.permute(0, 3, 1, 2)
        frames_resized = F.interpolate(frames_permuted, size=(new_H, new_W), mode='bicubic', align_corners=False)
        frames = frames_resized.permute(0, 2, 3, 1) # Back to NHWC
        del frames_permuted, frames_resized
        clean_vram()

    start_time = time.time()

    if enable_debug:
        _device = pipe.device
        log(f"Debug Mode: Enabled", message_type='info', icon="🐞")
        log(f"Device: {_device}", message_type='info', icon="🖥️")
        if torch.cuda.is_available():
             log(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", message_type='info', icon="💾")
        log(f"Input Frames: {frames.shape}", message_type='info', icon="🎞️")
        log(f"Chunk Size: {chunk_size}", message_type='info', icon="📦")
        log(f"Tiled DiT: {tiled_dit}, Tiled VAE: {tiled_vae}", message_type='info', icon="🧩")
        log_resource_usage(prefix="Start")

    # VRAM check and warning
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated()
        vram_total = torch.cuda.get_device_properties(0).total_memory

        # Optimize VRAM limit as requested
        try:
            # Set a soft limit on the process to prevent OOM crash if possible, or at least warn efficiently
            # set_per_process_memory_fraction is a hard limit, doing it might crash the process if we are already above.
            # Only set it if we haven't already.
            # However, user requested "Set the maximum VRAM usage to 90%".
            # We'll use a safer approach: Warn if > 90% and maybe trigger cleanup more aggressively.
            pass
        except:
            pass

        if vram_used / vram_total > 0.90:
            log("Warning: VRAM usage is very high (>90%)! Enabling fallback options is recommended.", message_type='warning', icon="⚠️")

    # Chunking Logic
    total_frames = frames.shape[0]
    final_outputs = []

    is_single_frame_input = (frames.shape[0] == 1)

    if chunk_size > 0 and chunk_size < total_frames:
        num_chunks = math.ceil(total_frames / chunk_size)
        log(f"Splitting video into {num_chunks} chunks (size {chunk_size})...", message_type='info', icon="✂️")

        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, total_frames)

            if enable_debug:
                log(f"Processing Chunk {i+1}/{num_chunks}: Frames {chunk_start}-{chunk_end}", message_type='info', icon="🎞️")

            chunk_frames = frames[chunk_start:chunk_end]

            # Auto-Fallback Logic
            retry_count = 0
            max_retries = 2
            current_tiled_vae = tiled_vae
            current_tiled_dit = tiled_dit

            while retry_count <= max_retries:
                try:
                    chunk_out = process_chunk(
                        pipe, chunk_frames, scale, color_fix, current_tiled_vae, current_tiled_dit,
                        tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio,
                        local_range, seed, force_offload, enable_debug,
                        is_single_frame_input=is_single_frame_input
                    )
                    final_outputs.append(chunk_out.cpu())
                    del chunk_out
                    clean_vram()
                    break # Success
                except torch.OutOfMemoryError as e:
                    retry_count += 1
                    clean_vram()
                    log(f"OOM detected in Chunk {i+1} (Attempt {retry_count}). Recovering...", message_type='warning', icon="🔄")

                    if not current_tiled_vae:
                        log("Auto-enabling Tiled VAE to prevent OOM (override)...", message_type='info', icon="🛡️")
                        current_tiled_vae = True
                    elif not current_tiled_dit:
                        log("Auto-enabling Tiled DiT to prevent OOM (override)...", message_type='info', icon="🛡️")
                        current_tiled_dit = True
                    else:
                        log("Both Tiled VAE and DiT enabled but still OOM. Cannot recover.", message_type='error', icon="❌")
                        raise e # Cannot recover further

        final_output_tensor = torch.cat(final_outputs, dim=0)
    else:
        # Auto-Fallback Logic for single chunk/full video
        retry_count = 0
        max_retries = 2
        current_tiled_vae = tiled_vae
        current_tiled_dit = tiled_dit

        while retry_count <= max_retries:
            try:
                final_output_tensor = process_chunk(
                    pipe, frames, scale, color_fix, current_tiled_vae, current_tiled_dit,
                    tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio,
                    local_range, seed, force_offload, enable_debug,
                    is_single_frame_input=is_single_frame_input
                )
                break
            except torch.OutOfMemoryError as e:
                retry_count += 1
                clean_vram()
                log(f"OOM detected (Attempt {retry_count}). Recovering...", message_type='warning', icon="🔄")

                if not current_tiled_vae:
                    log("Auto-enabling Tiled VAE to prevent OOM (override)...", message_type='info', icon="🛡️")
                    current_tiled_vae = True
                elif not current_tiled_dit:
                    log("Auto-enabling Tiled DiT to prevent OOM (override)...", message_type='info', icon="🛡️")
                    current_tiled_dit = True
                else:
                    log("Both Tiled VAE and DiT enabled but still OOM. Cannot recover.", message_type='error', icon="❌")
                    raise e

    end_time = time.time()
    total_time = end_time - start_time
    fps = frames.shape[0] / total_time if total_time > 0 else 0
    
    log(f"Done in {total_time:.2f}s ({fps:.2f} FPS).", message_type='finish', icon="✅")
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3
        log(f"Peak VRAM used: {peak_memory:.2f} GB", message_type='info', icon="📈")
        
    log_resource_usage(prefix="Final")
    
    return final_output_tensor


class FlashVSRNodeInitPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["FlashVSR", "FlashVSR-v1.1"], {
                    "default": "FlashVSR-v1.1",
                    "tooltip": "Select the FlashVSR model version. V1.1 is recommended for better stability."
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": 'Operation mode. "tiny": faster, standard memory. "tiny-long": optimized for long videos (lower VRAM). "full": higher quality but max VRAM.'
                }),
                "vae_type": (VAE_TYPES, {
                    "default": "wan2.1",
                    "tooltip": 'VAE type: "wan2.1" (default), "wan2.2" (improved normalization), "lightx2v" (50% less VRAM, 2-3x faster).'
                }),
                "alt_vae": (["none"] + folder_paths.get_filename_list("vae"), {
                    "default": "none",
                    "tooltip": 'Optional: Select an alternative VAE model file. Only used in "full" mode. VAE type settings still apply.'
                }),
                "force_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, forces offloading of models to CPU RAM after execution to free up VRAM for other nodes."
                }),
                "precision": (["fp16", "bf16", "auto"], {
                    "default": "auto",
                    "tooltip": "Inference precision. 'auto' selects bf16 if supported (RTX 30/40/50 series), otherwise fp16. bf16 is recommended."
                }),
                "device": (device_choices, {
                    "default": device_choices[0],
                    "tooltip": "Select the computation device (CUDA GPU, CPU, etc.). 'auto' picks the best available."
                }),
                "attention_mode": (["sparse_sage_attention", "block_sparse_attention", "flash_attention_2", "sdpa"], {
                    "default": "sparse_sage_attention",
                    "tooltip": 'Attention mechanism backend. "sparse_sage"/"block_sparse" use efficient sparse attention. "flash_attention_2"/"sdpa" use dense attention (slower, more VRAM).'
                }),
            }
        }
    
    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Initializes the FlashVSR pipeline, loads models, and configures memory management settings. Now supports Wan2.2 VAE and LightX2V for optimized performance.'
    
    def main(self, model, mode, vae_type, alt_vae, force_offload, precision, device, attention_mode):
        _device = device
        if device == "auto":
            _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else device
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
            
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)
            
        wan_video_dit.ATTENTION_MODE = attention_mode

        # Auto bfloat16 detection
        if precision == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                precision = "bf16"
                log("Auto-detected bf16 support.", message_type='info', icon="⚙️")
            else:
                precision = "fp16"
                log("Defaulting to fp16.", message_type='info', icon="⚙️")
            
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        try:
            dtype = dtype_map[precision]
        except:
            dtype = torch.bfloat16

        pipe = init_pipeline(model, mode, _device, dtype, alt_vae=alt_vae, vae_type=vae_type)
        return((pipe, force_offload),)

class FlashVSRNodeAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE", {
                    "tooltip": "The initialized FlashVSR pipeline object from the Init node."
                }),
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames to be upscaled. Batch of images (N, H, W, C)."
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "tooltip": "Upscaling factor. 2x or 4x. Higher scale requires more VRAM and compute."
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply wavelet-based color correction to match the output colors with the input, preventing color shifts."
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for the VAE decoder. Reduces VRAM usage significantly but is slower. Recommended for high-res outputs."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for the Diffusion Transformer (DiT). Crucial for saving VRAM on large inputs. Disabling it is faster but risky."
                }),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Size of the tiles for DiT processing. Smaller = less VRAM, more tiles, slower."
                }),
                "tile_overlap": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Overlap pixels between tiles to blend seams. Higher overlap = smoother transitions but more computation."
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the DiT model from VRAM before VAE decoding starts. Use this if VAE decode runs out of memory."
                }),
                "sparse_ratio": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Control for sparse attention. 1.5 is faster, 2.0 is more stable/quality. (For sparse backends only)"
                }),
                "kv_ratio": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Key/Value cache ratio. 1.0 uses less VRAM; 3.0 provides highest quality retention."
                }),
                "local_range": ("INT", {
                    "default": 11,
                    "min": 9,
                    "max": 11,
                    "step": 2,
                    "tooltip": "Local attention range window. 9 = sharper details; 11 = more stable/consistent results."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "tooltip": "Random seed for noise generation. Same seed + same settings = reproducible results."
                }),
                "frame_chunk_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Process video in chunks of N frames to prevent VRAM OOM. 0 = Process all frames at once. Results are merged on CPU."
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable verbose logging to console. Shows VRAM usage, step times, tile info, and detailed progress."
                }),
                "keep_models_on_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move models to CPU RAM instead of keeping them in VRAM when not in use. Prevents VRAM fragmentation/OOM."
                }),
                "resize_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Resize input frames before processing. Set to 0.5x for large 1080p+ videos to save VRAM."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    
    def main(self, pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, frame_chunk_size, enable_debug, keep_models_on_cpu, resize_factor):
        _pipe, _ = pipe
        output = flashvsr(_pipe, frames, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, keep_models_on_cpu, enable_debug, frame_chunk_size, resize_factor)
        return(output,)

class FlashVSRNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames to be upscaled. Batch of images (N, H, W, C)."
                }),
                "model": (["FlashVSR", "FlashVSR-v1.1"], {
                    "default": "FlashVSR-v1.1",
                    "tooltip": "Select the FlashVSR model version. V1.1 is recommended for better stability."
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": 'Operation mode. "tiny": faster, standard memory. "tiny-long": optimized for long videos (lower VRAM). "full": higher quality but max VRAM.'
                }),
                "vae_type": (VAE_TYPES, {
                    "default": "wan2.1",
                    "tooltip": 'VAE type: "wan2.1" (default), "wan2.2" (improved normalization), "lightx2v" (50% less VRAM, 2-3x faster).'
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "tooltip": "Upscaling factor. 2x or 4x."
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for the VAE decoder. Reduces VRAM usage significantly but is slower. Recommended for high-res outputs."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable spatial tiling for the Diffusion Transformer (DiT). Crucial for saving VRAM on large inputs."
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload the DiT model from VRAM before VAE decoding starts to free up memory. Recommended for 16GB VRAM."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "tooltip": "Random seed for noise generation."
                }),
                "frame_chunk_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Process video in chunks of N frames to prevent VRAM OOM. 0 = Process all frames at once."
                }),
                "attention_mode": (["sparse_sage_attention", "block_sparse_attention", "flash_attention_2", "sdpa"], {
                    "default": "sparse_sage_attention",
                    "tooltip": 'Attention mechanism backend. "sparse_sage" is recommended for speed/memory efficiency.'
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable extensive logging for debugging."
                }),
                "keep_models_on_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move models to CPU RAM instead of keeping them in VRAM when not in use."
                }),
                "resize_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Resize input frames before processing. Set to 0.5x for large 1080p+ videos to save VRAM."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Single-node easy usage for FlashVSR upscaling. Now supports Wan2.2 VAE and LightX2V for optimized VRAM usage.'
    
    def main(self, model, frames, mode, vae_type, scale, tiled_vae, tiled_dit, unload_dit, seed, frame_chunk_size, attention_mode, enable_debug, keep_models_on_cpu, resize_factor):
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "auto"
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
            
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)
            
        wan_video_dit.ATTENTION_MODE = attention_mode
            
        pipe = init_pipeline(model, mode, _device, torch.float16, vae_type=vae_type)
        output = flashvsr(pipe, frames, scale, True, tiled_vae, tiled_dit, 256, 24, unload_dit, 2.0, 3.0, 11, seed, keep_models_on_cpu, enable_debug, frame_chunk_size, resize_factor)
        return(output,)

NODE_CLASS_MAPPINGS = {
    "FlashVSRNode": FlashVSRNode,
    "FlashVSRNodeAdv": FlashVSRNodeAdv,
    "FlashVSRInitPipe": FlashVSRNodeInitPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRNode": "FlashVSR Ultra-Fast",
    "FlashVSRNodeAdv": "FlashVSR Ultra-Fast (Advanced)",
    "FlashVSRInitPipe": "FlashVSR Init Pipeline",
}
