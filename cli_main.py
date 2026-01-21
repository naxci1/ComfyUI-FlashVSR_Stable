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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Input video file path (e.g., video.mp4)'
    )
    input_group.add_argument(
        '--input_frames', '-if',
        type=str,
        help='Input folder containing extracted frames (e.g., frames/)'
    )
    
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--output', '-o',
        type=str,
        help='Output video file path (e.g., upscaled.mp4)'
    )
    output_group.add_argument(
        '--output_frames', '-of',
        type=str,
        help='Output folder for upscaled frames (e.g., upscaled_frames/)'
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
        '--quality',
        type=int,
        default=95,
        help='Output quality for frame images (1-100, higher = better quality). (default: 95)'
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

# =============================================================================
# Video I/O utilities (Stream based)
# =============================================================================

class VideoReader:
    """
    Iterator that reads video frames in chunks to save memory.
    """
    def __init__(self, video_path, start_frame=0, end_frame=-1, chunk_size=0):
        import cv2
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.chunk_size = chunk_size
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust end_frame
        if self.end_frame < 0 or self.end_frame > self.total_frames:
            self.end_frame = self.total_frames
            
        if self.start_frame >= self.total_frames:
            print(f"Warning: Start frame {self.start_frame} is beyond total frames {self.total_frames}.")
            self.end_frame = self.start_frame # Nothing to process

        self.current_frame = self.start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_frame >= self.end_frame:
            self.cap.release()
            raise StopIteration

        import torch
        import numpy as np
        import cv2

        frames = []
        frames_to_read = self.chunk_size if self.chunk_size > 0 else (self.end_frame - self.current_frame)
        
        # Ensure we don't read past end_frame
        frames_to_read = min(frames_to_read, self.end_frame - self.current_frame)
        
        if frames_to_read <= 0:
            self.cap.release()
            raise StopIteration

        for _ in range(frames_to_read):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_normalized)
            self.current_frame += 1

        if not frames:
            self.cap.release()
            raise StopIteration

        # Stack frames into tensor: (N, H, W, C)
        frames_tensor = torch.from_numpy(np.stack(frames, axis=0))
        return frames_tensor

    def get_info(self):
        return self.fps, self.total_frames

class VideoWriter:
    """
    Incremental video writer.
    """
    def __init__(self, output_path, fps, width, height, codec='libx264', crf=18):
        import cv2
        self.output_path = output_path
        self.width = width
        self.height = height
        self.codec = codec
        self.crf = crf
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Determine codec fourcc
        if codec in ['libx264', 'h264']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v is safer for general compatibility without ffmpeg specifically
        elif codec in ['libx265', 'hevc']:
            fourcc = cv2.VideoWriter_fourcc(*'hvc1')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
         
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.out.isOpened():
             print("Warning: cv2.VideoWriter failed to open with default flags. Trying 'mp4v'.")
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
             if not self.out.isOpened():
                 raise RuntimeError(f"Failed to create output video: {output_path}")

    def write(self, frames_tensor):
        import torch
        import numpy as np
        import cv2
        
        # Convert tensor to numpy
        if isinstance(frames_tensor, torch.Tensor):
            frames_np = frames_tensor.cpu().numpy()
        else:
            frames_np = frames_tensor
        
        # Ensure values are in [0, 1] and convert to uint8
        frames_np = np.clip(frames_np, 0.0, 1.0)
        frames_np = (frames_np * 255).astype(np.uint8)
        
        n_frames = frames_np.shape[0]
        
        for i in range(n_frames):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
            self.out.write(frame_bgr)
             
    def release(self):
        if self.out:
            self.out.release()

class FrameReader:
    """
    Iterator that reads image frames from a folder.
    """
    def __init__(self, frames_folder, start_frame=0, end_frame=-1, chunk_size=0):
        import numpy as np
        from PIL import Image
        import glob
        
        self.frames_folder = frames_folder
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.chunk_size = chunk_size
        self.current_frame_idx = 0
        self.current_frame = 0
        
        if not os.path.exists(frames_folder):
            raise FileNotFoundError(f"Input frames folder not found: {frames_folder}")
        
        # Get all image files and sort them
        supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
        frame_files = []
        for ext in supported_extensions:
            frame_files.extend(glob.glob(os.path.join(frames_folder, ext)))
        
        if not frame_files:
            raise RuntimeError(f"No image files found in folder: {frames_folder}")
        
        # Sort files by name to ensure consistent ordering
        frame_files.sort()
        self.frame_files = frame_files
        self.total_frames = len(frame_files)
        
        # Adjust end_frame
        if self.end_frame < 0 or self.end_frame > self.total_frames:
            self.end_frame = self.total_frames
            
        if self.start_frame >= self.total_frames:
            print(f"Warning: Start frame {self.start_frame} is beyond total frames {self.total_frames}.")
            self.end_frame = self.start_frame # Nothing to process
        
        self.current_frame_idx = self.start_frame

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_frame_idx >= self.end_frame:
            raise StopIteration
        
        import torch
        import numpy as np
        from PIL import Image
        
        frames = []
        frames_to_read = self.chunk_size if self.chunk_size > 0 else (self.end_frame - self.current_frame_idx)
        
        # Ensure we don't read past end_frame
        frames_to_read = min(frames_to_read, self.end_frame - self.current_frame_idx)
        
        if frames_to_read <= 0:
            raise StopIteration
        
        for _ in range(frames_to_read):
            if self.current_frame_idx >= len(self.frame_files):
                break
                
            frame_path = self.frame_files[self.current_frame_idx]
            
            # Read image using PIL
            img = Image.open(frame_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array and normalize to [0, 1]
            frame_np = np.array(img).astype(np.float32) / 255.0
            frames.append(frame_np)
            self.current_frame_idx += 1
        
        if not frames:
            raise StopIteration
        
        # Stack frames into tensor: (N, H, W, C)
        frames_tensor = torch.from_numpy(np.stack(frames, axis=0))
        return frames_tensor

    def get_info(self):
        # For frame processing, we don't have FPS information
        # Return a dummy FPS value and total frames
        return 30.0, self.total_frames

class FrameWriter:
    """
    Writes upscaled frames to a folder, preserving original names and formats.
    """
    def __init__(self, output_folder, quality=95):
        self.output_folder = output_folder
        self.quality = quality
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Store frame filenames for later use
        self.frame_filenames = []

    def set_frame_filenames(self, frame_filenames):
        """Set the original frame filenames to preserve naming."""
        self.frame_filenames = frame_filenames

    def write(self, frames_tensor, frame_indices=None):
        """
        Write frames to output folder.
        
        Args:
            frames_tensor: Tensor of shape (N, H, W, C) containing frames to write
            frame_indices: List of indices corresponding to original frame positions
        """
        import torch
        import numpy as np
        from PIL import Image
        
        # Convert tensor to numpy
        if isinstance(frames_tensor, torch.Tensor):
            frames_np = frames_tensor.cpu().numpy()
        else:
            frames_np = frames_tensor
        
        # Ensure values are in [0, 1] and convert to uint8
        frames_np = np.clip(frames_np, 0.0, 1.0)
        frames_np = (frames_np * 255).astype(np.uint8)
        
        n_frames = frames_np.shape[0]
        
        for i in range(n_frames):
            # Convert to PIL Image
            frame_pil = Image.fromarray(frames_np[i])
            
            # Determine output filename
            if frame_indices is not None and i < len(frame_indices):
                # Use original filename if available
                original_idx = frame_indices[i]
                if original_idx < len(self.frame_filenames):
                    original_path = self.frame_filenames[original_idx]
                    filename = os.path.basename(original_path)
                    output_path = os.path.join(self.output_folder, filename)
                else:
                    # Fallback naming
                    output_path = os.path.join(self.output_folder, f"frame_{original_idx:06d}.png")
            else:
                # Use sequential naming
                output_path = os.path.join(self.output_folder, f"frame_{self.current_frame:06d}.png")
                self.current_frame += 1
            
            # Save image with appropriate quality
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                frame_pil.save(output_path, quality=self.quality, subsampling=0)
            else:
                # For PNG and other formats, use default settings
                frame_pil.save(output_path)

    def release(self):
        # No resources to release for frame writer
        pass

def format_time(seconds):
    """
    Format seconds into HH:MM:SS or MM:SS
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


# =============================================================================
# Main CLI entry point
# =============================================================================

def main():
    args = parse_args()

    # Determine if we're processing frames or video
    is_frame_processing = hasattr(args, 'input_frames') and args.input_frames is not None
    
    # Safety check for output
    if not is_frame_processing:
        # Video processing - check if output file exists
        if os.path.exists(args.output):
            print(f"Error: Output file '{args.output}' already exists. Aborting to prevent overwrite.", file=sys.stderr)
            sys.exit(1)
    else:
        # Frame processing - check if output folder exists and is not empty
        if os.path.exists(args.output_frames):
            # Check if folder is empty
            if os.listdir(args.output_frames):
                print(f"Warning: Output folder '{args.output_frames}' already exists and is not empty.", file=sys.stderr)
                print("Existing files may be overwritten.", file=sys.stderr)

    # Handle boolean flag pairs
    force_offload = args.force_offload and not args.no_force_offload
    color_fix = args.color_fix and not args.no_color_fix
    keep_models_on_cpu = args.keep_models_on_cpu and not args.no_keep_models_on_cpu

    print("=" * 60)
    print("FlashVSR CLI - Video Super Resolution")
    print("=" * 60)
    
    if is_frame_processing:
        print(f"Input Frames: {args.input_frames}")
        print(f"Output Frames: {args.output_frames}")
        print(f"Quality: {args.quality}")
    else:
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
    
    print(f"Model: {args.model}, Mode: {args.mode}")
    print(f"VAE: {args.vae_model}, Scale: {args.scale}x")
    print("=" * 60)

    # ==========================================================================
    # Setup environment and imports
    # ==========================================================================
    
    # Mock ComfyUI modules for standalone CLI operation
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
    # Load input (Video or Frames)
    # ==========================================================================
    if is_frame_processing:
        print("\nInitializing Frame Reader...")
        reader = FrameReader(
            args.input_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            chunk_size=args.frame_chunk_size
        )
        
        # For frame processing, we use a dummy FPS value
        input_fps = 30.0
        file_total_frames = reader.total_frames
        
        # Calculate actual frames to process based on reader's resolved range
        total_frames_to_process = reader.end_frame - reader.start_frame
        
        if args.end_frame > 0 or args.start_frame > 0:
            print(f"Input: {args.input_frames} (folder)")
            print(f"Processing frames: {reader.start_frame} to {reader.end_frame} (Total: {total_frames_to_process})")
        else:
            print(f"Input: {args.input_frames} (folder, {total_frames_to_process} frames)")
    else:
        print("\nInitializing Video Reader...")
        reader = VideoReader(
            args.input,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            chunk_size=args.frame_chunk_size
        )

        input_fps, file_total_frames = reader.get_info()

        # Calculate actual frames to process based on reader's resolved range
        total_frames_to_process = reader.end_frame - reader.start_frame

        if args.end_frame > 0 or args.start_frame > 0:
            print(f"Input: {args.input} ({input_fps:.2f} FPS)")
            print(f"Processing frames: {reader.start_frame} to {reader.end_frame} (Total: {total_frames_to_process})")
        else:
            print(f"Input: {args.input} ({input_fps:.2f} FPS, {total_frames_to_process} frames)")
        
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
    # Process with FlashVSR (Chunk by Chunk)
    # ==========================================================================
    print("\nProcessing with FlashVSR...")

    writer = None
    total_processed = 0
    start_time_glob = 0

    try:
        import time
        start_time_glob = time.time()
        
        # For frame processing, we need to track frame indices for naming
        frame_indices = []
        
        for chunk_idx, frames in enumerate(reader):
            # Calculate progress metrics
            elapsed = time.time() - start_time_glob
            
            # Speed (fps) - avoid division by zero
            if total_processed > 0 and elapsed > 0:
                speed_fps = total_processed / elapsed
                remaining_frames = total_frames_to_process - total_processed
                eta_seconds = remaining_frames / speed_fps
            else:
                speed_fps = 0.0
                eta_seconds = 0
            
            formatted_elapsed = format_time(elapsed)
            formatted_eta = format_time(eta_seconds)
            
            # Print status for the *current state* (before processing this chunk)
            # format: Progress:   8.34% | Processed: 6464/77514 | Elapsed: 1:34:31 | ETA: 0:12:10 | Speed: 1.25 fps
            progress_pct = (total_processed / total_frames_to_process) * 100 if total_frames_to_process > 0 else 0
            print(f"Progress: {progress_pct:6.2f}% | Processed: {total_processed}/{total_frames_to_process} | "
                  f"Elapsed: {formatted_elapsed} | ETA: {formatted_eta} | Speed: {speed_fps:.2f} fps")
            
            # Process the chunk
            # Note: We pass chunk_size=0 to flashvsr because we are feeding it an explicit chunk
            # that we want processed fully right now.
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
                chunk_size=0, # Already chunked
                resize_factor=args.resize_factor,
                mode=args.mode
            )
            
            # Initialize Writer on first chunk
            if writer is None:
                h, w = output_frames.shape[1], output_frames.shape[2]
                print(f"Output dimensions: {w}x{h}")
                
                if is_frame_processing:
                    print(f"Saving output frames to: {args.output_frames}")
                    writer = FrameWriter(
                        output_folder=args.output_frames,
                        quality=args.quality
                    )
                    # Set frame filenames for frame writer
                    if hasattr(reader, 'frame_files'):
                        writer.set_frame_filenames(reader.frame_files)
                else:
                    print(f"Saving output video to: {args.output}")
                    writer = VideoWriter(
                        output_path=args.output,
                        fps=output_fps,
                        width=w,
                        height=h,
                        codec=args.codec,
                        crf=args.crf
                    )
            
            # Write frames
            if is_frame_processing:
                # For frame processing, track which original frames these correspond to
                start_idx = total_processed
                end_idx = total_processed + frames.shape[0]
                frame_indices = list(range(start_idx, end_idx))
                writer.write(output_frames, frame_indices)
            else:
                writer.write(output_frames)
            
            total_processed += frames.shape[0]
            
            # Cleanup
            del frames, output_frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except StopIteration:
        pass
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if writer:
            writer.release()
    
    # ==========================================================================
    # Cleanup
    # ==========================================================================
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    end_time_glob = time.time()
    total_duration = end_time_glob - start_time_glob
    avg_fps = total_processed / total_duration if total_duration > 0 else 0
    
    print("\n" + "=" * 60)
    if is_frame_processing:
        print("FlashVSR frame processing complete!")
    else:
        print("FlashVSR video processing complete!")
    print(f"Total Frames Processed: {total_processed}/{total_frames_to_process}")
    print(f"Total Time: {format_time(total_duration)} ({avg_fps:.2f} FPS)")
    print("=" * 60)


if __name__ == "__main__":
    main()
