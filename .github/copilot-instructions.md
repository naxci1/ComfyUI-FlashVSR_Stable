# GitHub Copilot Instructions for ComfyUI-FlashVSR_Stable

## Project Overview

ComfyUI-FlashVSR_Stable is a custom node for ComfyUI that enables video super-resolution (upscaling) using the FlashVSR model. The project is optimized for running on systems with limited VRAM (8GB-24GB+) and provides multiple operation modes to balance quality, speed, and memory usage.

### Key Features
- Video upscaling with 2x or 4x scale factors
- Multiple operation modes: `tiny`, `tiny-long`, and `full`
- Advanced VRAM optimization with tiling support for both VAE and DiT models
- Automatic OOM (Out of Memory) recovery with fallback mechanisms
- Chunked video processing for long videos
- Support for multiple attention backends (sparse_sage, flash_attention_2, etc.)
- Wavelet-based color correction to prevent color shifts

## Architecture

### Core Components

1. **Nodes** (`nodes.py`):
   - `FlashVSRNode`: Simple one-node interface for quick usage
   - `FlashVSRNodeAdv`: Advanced interface with full control over parameters
   - `FlashVSRNodeInitPipe`: Separate initialization node for pipeline reuse

2. **Pipelines** (`src/pipelines/`):
   - `FlashVSRFullPipeline`: Uses full VAE encoder, highest quality but most VRAM
   - `FlashVSRTinyPipeline`: Fast mode with TCDecoder instead of full VAE
   - `FlashVSRTinyLongPipeline`: Streaming mode for very long videos, lowest VRAM

3. **Models** (`src/models/`):
   - `wan_video_dit.py`: Diffusion Transformer model
   - `wan_video_vae.py`: Video VAE encoder/decoder
   - `TCDecoder.py`: Temporal-Causal decoder for tiny modes
   - `model_manager.py`: Centralized model loading and management
   - `utils.py`: VRAM management utilities

4. **VRAM Management** (`src/vram_management/`):
   - Advanced memory management for efficient GPU usage
   - Model offloading and loading strategies

## Code Conventions

### Python Style
- Follow PEP 8 style guidelines
- Use 4 spaces for indentation (not tabs)
- Use descriptive variable names that reflect the domain (e.g., `tiled_vae`, `unload_dit`, `frame_chunk_size`)

### Imports
- Group imports in order: standard library, third-party packages, local modules
- Use relative imports for internal modules (e.g., `from .src import ModelManager`)
- Provide fallback imports for ComfyUI integration (see `nodes.py` lines 18-29)

### Logging and User Feedback
- Use the custom `log()` function with appropriate message types: `'normal'`, `'error'`, `'warning'`, `'finish'`, `'info'`
- Include emoji icons for better readability (e.g., `icon="🚀"`, `icon="⚠️"`, `icon="✅"`)
- For progress tracking, use the custom `cqdm` class which wraps ComfyUI's ProgressBar
- Log resource usage (RAM/VRAM) at key points using `log_resource_usage()`

### VRAM Management Best Practices
- **Always** call `clean_vram()` after operations that free up tensors
- Use `torch.cuda.reset_peak_memory_stats()` at the start of major operations
- Delete large tensors explicitly with `del` before cleaning VRAM
- Offload models to CPU when not in use (controlled by `keep_models_on_cpu` parameter)
- Use tiling (`tiled_vae`, `tiled_dit`) for large resolutions to prevent OOM

### Error Handling
- Implement automatic OOM recovery with fallback options (see `flashvsr()` function)
- Retry with progressively more conservative settings: first enable `tiled_vae`, then `tiled_dit`
- Always provide meaningful error messages indicating what failed and potential solutions
- Wrap model loading in try-except blocks with fallback mechanisms (see lines 224-244)

### Tensor Operations
- Prefer native PyTorch operations over `einops` where performance matters
- Use `permute()` for dimension reordering instead of excessive rearranging
- Maintain NHWC format (Batch, Height, Width, Channels) for consistency with ComfyUI
- Internal processing uses NCHW or BCFHW (Batch, Channels, Frames, Height, Width) format

### Type Hints and Documentation
- Use type hints for function parameters and return values
- Add `tooltip` documentation for all node INPUT_TYPES parameters
- Document complex algorithms with inline comments
- Include parameter descriptions in node definitions

## Common Patterns

### Loading Models
```python
# Always use ModelManager for centralized model loading
mm = ModelManager(torch_dtype=dtype, device="cpu")
mm.load_models([ckpt_path, vae_path])
pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)

# Manual fallback for VAE loading if ModelManager fails
if pipe.vae is None:
    pipe.vae = WanVideoVAE(z_dim=16, dim=96).to(device=device, dtype=dtype)
    sd = torch.load(vae_path, map_location="cpu", weights_only=False)
    pipe.vae.load_state_dict(sd)
```

### Tiled Processing
```python
# Calculate tile coordinates with overlap for seamless blending
tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
final_output_canvas = torch.zeros((N, H * scale, W * scale, C), dtype=torch.float16, device="cpu")
weight_sum_canvas = torch.zeros_like(final_output_canvas)

# Process each tile with feather mask blending
for x1, y1, x2, y2 in tile_coords:
    tile = process_tile(x1, y1, x2, y2)
    mask = create_feather_mask((tile.shape[1], tile.shape[2]), tile_overlap * scale)
    # Accumulate weighted results
    final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += tile * mask
    weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask
```

### Frame Preparation
```python
# Pad frame count to meet model requirements (8n+1 for full pipeline)
F = largest_8n1_leq(num_frames_with_padding)

# Upscale input frames to match target dimensions (multiple of 128)
sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)

# Normalize to [-1, 1] range for model input
tensor_normalized = tensor * 2.0 - 1.0
```

## Testing

### Running Tests
The project includes basic unit tests in `test_mock.py`:
```bash
python test_mock.py
```

**Note**: Tests mock ComfyUI dependencies since they're not available in standard environments.

### Manual Testing
1. Install ComfyUI and clone this repo into `ComfyUI/custom_nodes/`
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Download FlashVSR models to `ComfyUI/models/FlashVSR/`
4. Load the sample workflow from `workflow/FlashVSR.json`
5. Test with different VRAM configurations and video lengths

## Dependencies

### Core Requirements
- PyTorch (with CUDA support for GPU acceleration)
- ComfyUI framework (custom_nodes integration)
- `huggingface_hub` for model downloading
- `safetensors` for model weight loading
- `einops` for tensor operations
- `psutil` for system resource monitoring

### Optional Performance Libraries
- `sageattention` for efficient sparse attention (recommended)
- `flash-attn` for flash attention (CUDA only)
- `triton` / `triton-windows` for kernel optimization (version <3.3.0 for Turing/older GPUs)

## VRAM Optimization Strategies

### Memory Hierarchy (by VRAM usage, ascending)
1. **tiny-long mode + tiling + chunking** (8GB VRAM)
2. **tiny mode + tiling** (12GB VRAM)
3. **tiny mode + tiled_vae only** (16GB VRAM)
4. **tiny mode without tiling** (20GB+ VRAM)
5. **full mode without tiling** (24GB+ VRAM)

### Key Parameters for VRAM Control
- `tiled_vae`: Spatial tiling for VAE decoder (significant VRAM savings)
- `tiled_dit`: Spatial tiling for DiT model (critical for large resolutions)
- `tile_size`: Smaller = less VRAM, more tiles, slower processing
- `tile_overlap`: Higher = smoother seams but more computation
- `unload_dit`: Unload DiT before VAE decode (helps when VAE OOMs)
- `frame_chunk_size`: Process video in chunks, offload to CPU between chunks
- `resize_factor`: Reduce input resolution before processing (e.g., 0.5x for 1080p)
- `keep_models_on_cpu`: Move models to RAM when idle

### Automatic OOM Recovery
The `flashvsr()` function implements automatic fallback:
1. Try with user-specified settings
2. On OOM: retry with `tiled_vae=True`
3. Still OOM: retry with both `tiled_vae=True` and `tiled_dit=True`
4. If still OOM: raise error (cannot recover)

## Common Pitfalls and Gotchas

### 1. Frame Count Padding
- Model requires specific frame counts: 8n+1 for full, 8n+5 for tiny-long
- Always pad frames by repeating the last frame to meet requirements
- Don't forget to crop output back to original frame count

### 2. Dimension Alignment
- Internal processing requires dimensions to be multiples of 128
- Use `compute_scaled_and_target_dims()` to calculate correct dimensions
- Crop output back to expected dimensions after processing

### 3. Tensor Device Management
- Tiny-long mode keeps LQ video on CPU, loads chunks to GPU as needed
- Other modes move entire LQ video to GPU
- Always move final output to CPU to free VRAM

### 4. Model Loading
- Some model files use `weights_only=False` due to legacy format (lines 238, 256, 263)
- Always check if ModelManager successfully loaded models (especially VAE in full mode)
- Implement fallback loading mechanisms for robustness

### 5. Progress Bar Integration
- Use `cqdm` wrapper to integrate with ComfyUI's ProgressBar
- For loops without return values, use context manager: `with cqdm(...)`
- For iterables, use as iterator: `for item in cqdm(...)`
- Pass `enable_debug` to show detailed logs and resource usage

### 6. Attention Mode Compatibility
- `sparse_sage_attention` and `block_sparse_attention` require sageattention package
- `flash_attention_2` requires flash-attn package (CUDA only, not Windows by default)
- `sdpa` is PyTorch native but slower/more memory
- Set via `wan_video_dit.ATTENTION_MODE` before pipeline initialization

### 7. Precision Handling
- Auto-detect bf16 support: `torch.cuda.is_bf16_supported()`
- bf16 recommended for RTX 30/40/50 series (better dynamic range than fp16)
- fp16 fallback for older GPUs
- Some operations internally use fp16 (e.g., tiled output canvas)

### 8. Color Correction
- `color_fix=True` applies wavelet-based color transfer from input to output
- Prevents color shifts that can occur during upscaling
- Highly recommended to keep enabled unless you have specific reasons

## File Organization

```
ComfyUI-FlashVSR_Stable/
├── .github/
│   └── copilot-instructions.md    # This file
├── nodes.py                        # Main ComfyUI node definitions
├── __init__.py                     # Package initialization, exports NODE_CLASS_MAPPINGS
├── requirements.txt                # Python dependencies
├── posi_prompt.pth                # Positive prompt embeddings for model
├── test_mock.py                   # Unit tests with mocked dependencies
├── src/
│   ├── __init__.py               # Exports models, pipelines, schedulers
│   ├── models/                   # Model implementations
│   │   ├── model_manager.py     # Centralized model loading
│   │   ├── wan_video_dit.py     # Diffusion Transformer
│   │   ├── wan_video_vae.py     # Video VAE
│   │   ├── TCDecoder.py         # Temporal-Causal Decoder
│   │   ├── utils.py             # VRAM utilities
│   │   └── sparse_sage/         # Sparse attention implementations
│   ├── pipelines/               # Processing pipelines
│   │   ├── base.py             # Base pipeline class
│   │   ├── flashvsr_full.py    # Full mode pipeline
│   │   ├── flashvsr_tiny.py    # Tiny mode pipeline
│   │   └── flashvsr_tiny_long.py # Tiny-long streaming pipeline
│   ├── schedulers/             # Noise schedulers
│   └── vram_management/        # Memory management utilities
├── workflow/                   # Example ComfyUI workflows
└── img/                       # Documentation images
```

## When Making Changes

### Adding New Features
1. Consider VRAM impact - will this increase memory usage?
2. Add appropriate parameters to node INPUT_TYPES with tooltips
3. Update README.md with new feature documentation
4. Test on different VRAM tiers (8GB, 12GB, 16GB, 24GB+)
5. Implement fallback behavior for OOM scenarios

### Optimizing Performance
1. Profile with `enable_debug=True` to identify bottlenecks
2. Check if operations can be batched or parallelized
3. Consider moving operations to CPU if they're memory-bound
4. Use native PyTorch ops instead of einops where possible
5. Minimize data transfers between CPU and GPU

### Bug Fixes
1. Check if issue is VRAM-related (most common)
2. Verify tensor shapes at each operation (print if needed)
3. Ensure proper device placement (CPU vs GPU)
4. Test with different modes (tiny, tiny-long, full)
5. Validate with edge cases (single frame, very long videos, odd dimensions)

### Updating Dependencies
1. Check compatibility with ComfyUI version
2. Test on both Windows and Linux if possible
3. Verify triton version constraints for older GPUs
4. Update requirements.txt with version pins if needed

## Integration with ComfyUI

### Node Registration
Nodes are registered via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`:
```python
NODE_CLASS_MAPPINGS = {
    "FlashVSRNode": FlashVSRNode,
    "FlashVSRNodeAdv": FlashVSRNodeAdv,
    "FlashVSRInitPipe": FlashVSRNodeInitPipe,
}
```

### Input/Output Types
- **IMAGE**: ComfyUI tensor format (Batch, Height, Width, Channels) in [0, 1] range
- **PIPE**: Custom tuple containing (pipeline_object, force_offload_flag)
- **INT**, **FLOAT**, **BOOLEAN**: Standard parameter types

### Progress Reporting
- Use `comfy.utils.ProgressBar(total)` for progress tracking
- Wrap with `cqdm` to add console logging
- Update progress with `pbar.update(1)` for each iteration

## Performance Tips

1. **For 8GB VRAM**: Use tiny-long mode, enable all tiling, set chunk_size=20, resize_factor=0.5
2. **For 12GB VRAM**: Use tiny mode, enable tiling, sparse_sage attention
3. **For 16GB VRAM**: Use tiny mode, tiled_vae=True, set unload_dit=True
4. **For 24GB+ VRAM**: Use full or tiny mode, disable tiling for max speed
5. **For long videos (>100 frames)**: Always use frame_chunk_size to prevent VRAM saturation
6. **For 1080p+ input**: Set resize_factor=0.5 to reduce input dimensions before upscaling

## Useful References

- [FlashVSR Paper](https://arxiv.org/abs/2410.13115) - Original research paper
- [FlashVSR GitHub](https://github.com/OpenImagingLab/FlashVSR) - Original implementation
- [ComfyUI Documentation](https://docs.comfy.org/) - ComfyUI custom node development
- [Sparse SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) - Efficient attention mechanism
