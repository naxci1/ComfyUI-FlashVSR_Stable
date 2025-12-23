#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR Command-Line Interface
===============================

A mirror-grade CLI that maps 1:1 with the ComfyUI node inputs.
All parameters from FlashVSRNode, FlashVSRNodeAdv, and FlashVSRNodeInitPipe
are exposed as command-line arguments.

Usage:
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2

For full help:
    python cli_main.py --help
"""

import argparse
import os
import sys
import gc

# =============================================================================
# CLI argument parsing - EXHAUSTIVE mapping from ComfyUI node INPUT_TYPES
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.
    
    Every argument corresponds directly to a parameter in the ComfyUI node
    INPUT_TYPES (FlashVSRNode, FlashVSRNodeAdv, FlashVSRNodeInitPipe).
    """
    parser = argparse.ArgumentParser(
        description="FlashVSR CLI - Video Super Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic 2x upscale with defaults
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2

    # 4x upscale with tiling enabled for lower VRAM
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 4 \\
        --tiled_vae --tiled_dit --tile_size 256 --tile_overlap 24

    # Long video with chunking to prevent OOM
    python cli_main.py --input long_video.mp4 --output upscaled.mp4 \\
        --frame_chunk_size 50 --mode tiny-long

    # Low VRAM mode (8GB GPUs)
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2 \\
        --vae_model LightVAE_W2.1 --tiled_vae --tiled_dit \\
        --frame_chunk_size 20 --resize_factor 0.5

For more information, visit: https://github.com/naxci1/ComfyUI-FlashVSR_Stable
"""
    )

    # ==========================================================================
    # Required arguments
    # ==========================================================================
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input video file path (e.g., video.mp4)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output video file path (e.g., upscaled.mp4)'
    )

    # ==========================================================================
    # FlashVSRNodeInitPipe parameters (Pipeline Initialization)
    # ==========================================================================
    parser.add_argument(
        '--model',
        type=str,
        choices=['FlashVSR', 'FlashVSR-v1.1'],
        default='FlashVSR-v1.1',
        help='FlashVSR model version. V1.1 is recommended for better stability. (default: FlashVSR-v1.1)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['tiny', 'tiny-long', 'full'],
        default='tiny',
        help='Operation mode. "tiny": faster, standard memory. "tiny-long": optimized for long videos (lower VRAM). "full": higher quality but max VRAM. (default: tiny)'
    )
    parser.add_argument(
        '--vae_model',
        type=str,
        choices=['Wan2.1', 'Wan2.2', 'LightVAE_W2.1', 'TAE_W2.2', 'LightTAE_HY1.5'],
        default='Wan2.1',
        help='VAE model: Wan2.1 (default), Wan2.2, LightVAE_W2.1 (50%% less VRAM), TAE_W2.2, LightTAE_HY1.5. Auto-downloads if missing. (default: Wan2.1)'
    )
    parser.add_argument(
        '--force_offload',
        action='store_true',
        default=True,
        help='Force offloading of models to CPU RAM after execution to free up VRAM. (default: True)'
    )
    parser.add_argument(
        '--no_force_offload',
        action='store_true',
        help='Disable force offloading (keeps models in VRAM).'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp16', 'bf16', 'auto'],
        default='auto',
        help="Inference precision. 'auto' selects bf16 if supported (RTX 30/40/50 series), otherwise fp16. (default: auto)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Computation device (e.g., "cuda:0", "cuda:1", "cpu", "auto"). (default: auto)'
    )
    parser.add_argument(
        '--attention_mode',
        type=str,
        choices=['sparse_sage_attention', 'block_sparse_attention', 'flash_attention_2', 'sdpa'],
        default='sparse_sage_attention',
        help='Attention mechanism backend. "sparse_sage"/"block_sparse" use efficient sparse attention. "flash_attention_2"/"sdpa" use dense attention. (default: sparse_sage_attention)'
    )

    # ==========================================================================
    # FlashVSRNodeAdv parameters (Processing)
    # ==========================================================================
    parser.add_argument(
        '--scale',
        type=int,
        choices=[2, 4],
        default=2,
        help='Upscaling factor. 2x or 4x. Higher scale requires more VRAM and compute. (default: 2)'
    )
    parser.add_argument(
        '--color_fix',
        action='store_true',
        default=True,
        help='Apply wavelet-based color correction to match output colors with input. (default: True)'
    )
    parser.add_argument(
        '--no_color_fix',
        action='store_true',
        help='Disable color correction.'
    )
    parser.add_argument(
        '--tiled_vae',
        action='store_true',
        default=False,
        help='Enable spatial tiling for the VAE decoder. Reduces VRAM usage significantly but is slower.'
    )
    parser.add_argument(
        '--tiled_dit',
        action='store_true',
        default=False,
        help='Enable spatial tiling for the Diffusion Transformer (DiT). Crucial for saving VRAM on large inputs.'
    )
    parser.add_argument(
        '--tile_size',
        type=int,
        default=256,
        help='Size of the tiles for DiT processing (32-1024). Smaller = less VRAM, more tiles, slower. (default: 256)'
    )
    parser.add_argument(
        '--tile_overlap',
        type=int,
        default=24,
        help='Overlap pixels between tiles to blend seams (8-512). Higher = smoother transitions. (default: 24)'
    )
    parser.add_argument(
        '--unload_dit',
        action='store_true',
        default=False,
        help='Unload the DiT model from VRAM before VAE decoding starts. Use if VAE decode runs out of memory.'
    )
    parser.add_argument(
        '--sparse_ratio',
        type=float,
        default=2.0,
        help='Control for sparse attention (1.5-2.0). 1.5 is faster, 2.0 is more stable/quality. (default: 2.0)'
    )
    parser.add_argument(
        '--kv_ratio',
        type=float,
        default=3.0,
        help='Key/Value cache ratio (1.0-3.0). 1.0 uses less VRAM; 3.0 provides highest quality retention. (default: 3.0)'
    )
    parser.add_argument(
        '--local_range',
        type=int,
        choices=[9, 11],
        default=11,
        help='Local attention range window. 9 = sharper details; 11 = more stable/consistent results. (default: 11)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for noise generation. Same seed + same settings = reproducible results. (default: 0)'
    )
    parser.add_argument(
        '--frame_chunk_size',
        type=int,
        default=0,
        help='Process video in chunks of N frames to prevent VRAM OOM. 0 = Process all frames at once. (default: 0)'
    )
    parser.add_argument(
        '--enable_debug',
        action='store_true',
        default=False,
        help='Enable verbose logging to console. Shows VRAM usage, step times, tile info, and detailed progress.'
    )
    parser.add_argument(
        '--keep_models_on_cpu',
        action='store_true',
        default=True,
        help='Move models to CPU RAM instead of keeping them in VRAM when not in use. (default: True)'
    )
    parser.add_argument(
        '--no_keep_models_on_cpu',
        action='store_true',
        help='Keep models in VRAM (faster but uses more VRAM).'
    )
    parser.add_argument(
        '--resize_factor',
        type=float,
        default=1.0,
        help='Resize input frames before processing (0.1-1.0). Set to 0.5 for large 1080p+ videos. (default: 1.0)'
    )

    # ==========================================================================
    # Video I/O parameters
    # ==========================================================================
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Output video FPS. If not specified, uses input video FPS.'
    )
    parser.add_argument(
        '--codec',
        type=str,
        default='libx264',
        help='Video codec for output (e.g., libx264, libx265, h264_nvenc). (default: libx264)'
    )
    parser.add_argument(
        '--crf',
        type=int,
        default=18,
        help='Constant Rate Factor for quality (0-51, lower = better quality). (default: 18)'
    )
    parser.add_argument(
        '--start_frame',
        type=int,
        default=0,
        help='Start processing from this frame index (0-indexed). (default: 0)'
    )
    parser.add_argument(
        '--end_frame',
        type=int,
        default=-1,
        help='Stop processing at this frame index (-1 = process all). (default: -1)'
    )

    # ==========================================================================
    # Model paths (optional, for custom model locations)
    # ==========================================================================
    parser.add_argument(
        '--models_dir',
        type=str,
        default=None,
        help='Custom path to FlashVSR models directory. If not set, uses ComfyUI default or ./models'
    )

    return parser.parse_args()


# =============================================================================
# Video I/O utilities
# =============================================================================

def load_video_frames(video_path, start_frame=0, end_frame=-1):
    """
    Load video frames from a file.
    
    Returns:
        frames: torch.Tensor of shape (N, H, W, C) with values in [0, 1]
        fps: float, original video FPS
    """
    import torch
    import numpy as np
    
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for video loading. Install with: pip install opencv-python")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end_frame < 0 or end_frame > total_frames:
        end_frame = total_frames
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= start_frame and frame_idx < end_frame:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_normalized)
        
        frame_idx += 1
        if frame_idx >= end_frame:
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise RuntimeError(f"No frames loaded from video: {video_path}")
    
    # Stack frames into tensor: (N, H, W, C)
    frames_tensor = torch.from_numpy(np.stack(frames, axis=0))
    
    print(f"Loaded {len(frames)} frames from {video_path} ({fps:.2f} FPS)")
    print(f"Frame dimensions: {frames_tensor.shape[1]}x{frames_tensor.shape[2]}")
    
    return frames_tensor, fps


def save_video_frames(frames_tensor, output_path, fps, codec='libx264', crf=18):
    """
    Save video frames to a file.
    
    Args:
        frames_tensor: torch.Tensor of shape (N, H, W, C) with values in [0, 1]
        output_path: Output video file path
        fps: Output FPS
        codec: Video codec
        crf: Constant Rate Factor
    """
    import torch
    import numpy as np
    
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for video saving. Install with: pip install opencv-python")

    # Convert tensor to numpy
    if isinstance(frames_tensor, torch.Tensor):
        frames_np = frames_tensor.cpu().numpy()
    else:
        frames_np = frames_tensor
    
    # Ensure values are in [0, 1] and convert to uint8
    frames_np = np.clip(frames_np, 0.0, 1.0)
    frames_np = (frames_np * 255).astype(np.uint8)
    
    n_frames, height, width, channels = frames_np.shape
    
    # Determine codec fourcc
    if codec in ['libx264', 'h264']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif codec in ['libx265', 'hevc']:
        fourcc = cv2.VideoWriter_fourcc(*'hvc1')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_path}")
    
    for i in range(n_frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    print(f"Saved {n_frames} frames to {output_path} ({fps:.2f} FPS)")
    print(f"Output dimensions: {width}x{height}")


# =============================================================================
# Main CLI entry point
# =============================================================================

def main():
    args = parse_args()
    
    # Handle boolean flag pairs
    force_offload = args.force_offload and not args.no_force_offload
    color_fix = args.color_fix and not args.no_color_fix
    keep_models_on_cpu = args.keep_models_on_cpu and not args.no_keep_models_on_cpu

    print("=" * 60)
    print("FlashVSR CLI - Video Super Resolution")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}, Mode: {args.mode}")
    print(f"VAE: {args.vae_model}, Scale: {args.scale}x")
    print("=" * 60)

    # ==========================================================================
    # Setup environment and imports
    # ==========================================================================
    
    # Mock ComfyUI modules for standalone CLI operation
    import sys
    from unittest.mock import MagicMock
    
    # Create mock folder_paths module
    folder_paths_mock = MagicMock()
    if args.models_dir:
        folder_paths_mock.models_dir = args.models_dir
    else:
        # Default to ./models or ComfyUI default
        folder_paths_mock.models_dir = os.path.join(os.path.dirname(__file__), "models")
    folder_paths_mock.get_filename_list = MagicMock(return_value=[])
    sys.modules['folder_paths'] = folder_paths_mock
    
    # Create mock comfy modules
    comfy_mock = MagicMock()
    comfy_utils_mock = MagicMock()
    comfy_utils_mock.ProgressBar = MagicMock()
    sys.modules['comfy'] = comfy_mock
    sys.modules['comfy.utils'] = comfy_utils_mock
    
    # Now import FlashVSR modules
    import torch
    
    # Set device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Device: {device}")
    
    # Import FlashVSR modules after mocking
    from nodes import (
        init_pipeline, flashvsr, log,
        VAE_MODEL_OPTIONS, VAE_MODEL_MAP
    )
    from src.models import wan_video_dit
    
    # ==========================================================================
    # Load input video
    # ==========================================================================
    print("\nLoading input video...")
    frames, input_fps = load_video_frames(
        args.input, 
        start_frame=args.start_frame, 
        end_frame=args.end_frame
    )
    
    # Use output FPS if specified, otherwise use input FPS
    output_fps = args.fps if args.fps is not None else input_fps
    
    # ==========================================================================
    # Initialize pipeline
    # ==========================================================================
    print("\nInitializing FlashVSR pipeline...")
    
    # Set attention mode
    wan_video_dit.ATTENTION_MODE = args.attention_mode
    
    # Determine dtype
    if args.precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Auto-detected bf16 support.")
        else:
            dtype = torch.float16
            print("Defaulting to fp16.")
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    # Set CUDA device if using CUDA
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    
    # Initialize the pipeline
    pipe = init_pipeline(
        model=args.model,
        mode=args.mode,
        device=device,
        dtype=dtype,
        vae_model=args.vae_model
    )
    
    # ==========================================================================
    # Process video with FlashVSR
    # ==========================================================================
    print("\nProcessing video with FlashVSR...")
    
    output_frames = flashvsr(
        pipe=pipe,
        frames=frames,
        scale=args.scale,
        color_fix=color_fix,
        tiled_vae=args.tiled_vae,
        tiled_dit=args.tiled_dit,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        unload_dit=args.unload_dit,
        sparse_ratio=args.sparse_ratio,
        kv_ratio=args.kv_ratio,
        local_range=args.local_range,
        seed=args.seed,
        force_offload=keep_models_on_cpu,  # flashvsr() uses force_offload param for CPU offloading
        enable_debug=args.enable_debug,
        chunk_size=args.frame_chunk_size,
        resize_factor=args.resize_factor,
        mode=args.mode
    )
    
    # ==========================================================================
    # Save output video
    # ==========================================================================
    print("\nSaving output video...")
    save_video_frames(
        frames_tensor=output_frames,
        output_path=args.output,
        fps=output_fps,
        codec=args.codec,
        crf=args.crf
    )
    
    # ==========================================================================
    # Cleanup
    # ==========================================================================
    del pipe, frames, output_frames
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("FlashVSR processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
