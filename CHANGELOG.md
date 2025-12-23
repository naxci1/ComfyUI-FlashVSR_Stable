# Changelog

All notable changes to ComfyUI-FlashVSR_Stable will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ✨ New Features

- **Full-Featured CLI**: Added `cli_main.py` with exhaustive command-line argument mapping that mirrors all ComfyUI node inputs 1:1. Supports all parameters from FlashVSRNodeInitPipe, FlashVSRNodeAdv, and FlashVSRNode including:
  - Pipeline initialization: `--model`, `--mode`, `--vae_model`, `--precision`, `--device`, `--attention_mode`
  - Processing controls: `--scale`, `--color_fix`, `--tiled_vae`, `--tiled_dit`, `--tile_size`, `--tile_overlap`
  - Advanced tuning: `--sparse_ratio`, `--kv_ratio`, `--local_range`, `--seed`, `--frame_chunk_size`
  - VRAM optimization: `--unload_dit`, `--keep_models_on_cpu`, `--resize_factor`
  - Video I/O: `--fps`, `--codec`, `--crf`, `--start_frame`, `--end_frame`, `--models_dir`

### 📖 Documentation

- Added comprehensive CLI arguments reference table to README.md with all parameter descriptions

### 🛠 Dependencies

- Added `opencv-python>=4.8.1.78` to requirements.txt for CLI video I/O support

---

## [1.2.0] - 2025-12-23

### ✨ New Features

- **5 New VAE Models**: Added support for `Wan2.2`, `LightVAE_W2.1`, `TAE_W2.2`, `LightTAE_HY1.5` (plus original `Wan2.1`).
- **Smart Resource Calculator**: Added "Pre-flight" check to analyze VRAM/RAM usage and recommend optimal settings before processing.
- **Auto-Download**: Added automatic downloading for missing VAE models from HuggingFace (`lightx2v/Autoencoders` repository).
- **Unified UI**: Replaced complex `vae_type` and `alt_vae` inputs with a single `vae_model` dropdown.

### 🐛 Bug Fixes

- **Fixed Black Borders**: Implemented correct "Padding → Process → Crop" logic to handle VAE dimension requirements without corrupting the output.
- **Fixed Video Glitches**: Corrected Tensor Permutation logic (`B,C,F,H,W` vs `B,H,W,C`) to prevent visual artifacts.
- **Fixed Model Loading**: Solved the issue where selecting "Wan2.2" incorrectly loaded "Wan2.1_VAE.pth".
- **Fixed OOM False Positive**: OOM recovery no longer triggers prematurely (adjusted threshold to 95%).

### ⚡ Optimization

- **Unified Pipeline**: Merged `tiny`, `full`, and `tiny-long` logic into a single optimized architecture.
- **OOM Protection**: Improved memory management with 95% VRAM threshold using `torch.cuda.mem_get_info()`.
- **Lossless Resize**: Uses `NEAREST` interpolation for integer scaling (0.5, 0.25) to avoid blur.
- **Aggressive Garbage Collection**: Added `torch.cuda.empty_cache()` before/after heavy VAE operations.

### 🛠 Refactoring

- **Explicit VAE Instantiation**: No more state_dict inspection/guessing - strict class mapping based on user selection.
- **VAE_MODEL_MAP**: Centralized configuration for all 5 VAE options (class, file, URL, dimensions).
- **Summary Logging**: End-of-processing report with total time, peak VRAM, and resolutions.
- **Debug Logging**: Shows `selected_model` vs `loaded_model_path` for verification.

### 📖 Documentation

- Updated README with VAE Selection Guide and VRAM tier recommendations.
- Added Best Practices section for Low/Medium/High VRAM configurations.
- Added Pre-Flight Resource Check documentation with example output.
- Added direct download links for all 5 VAE models.

---

## [1.1.0] - 2025-12-22

### 🚀 New Features

- **Wan2.2 VAE Support**: Integrated Wan2.2 VAE with optimized normalization statistics.
- **LightX2V VAE Integration**: Added LightX2V VAE for ~50% VRAM reduction and 2-3x faster inference.
- **VAE Type Selection**: Added `vae_model` dropdown in both Init Pipeline and Ultra-Fast nodes.
- **Factory Function**: Added `create_video_vae()` for programmatic VAE selection.

### ⚡ Performance

- **VRAM Reduction**: LightX2V reduces peak VRAM usage by approximately 50%.
- **Speed Improvement**: LightX2V provides 2-3x faster VAE decode times.

### 🛠 Refactoring

- **Backward Compatibility**: All new VAE types maintain full compatibility with existing Wan2.1 weights.
- **Architecture Constants**: Added `VAE_FULL_DIM`, `VAE_LIGHT_DIM`, `VAE_Z_DIM` for clarity.

---

## [1.0.3] - 2025-12-08

### ⚡ Performance

- **tiny-long Optimization**: Ported all VRAM optimizations and Tiled VAE support to `tiny-long` mode.
- **Windows Optimization**: Speed improvements for Windows environments.
- **Codebase Cleanup**: General synchronization and cleanup.

---

## [1.0.2] - 2025-12-07

### 🚀 New Features

- **Frame Chunking**: Added `frame_chunk_size` option to split large videos into chunks.
- **Attention Mode Selection**: Support for `flash_attention_2`, `sdpa`, `sparse_sage`, and `block_sparse`.
- **Debug Mode**: Added `enable_debug` option for extensive logging.

### 🐛 Bug Fixes

- **OOM Auto-Recovery**: If OOM occurs, automatically retries with `tiled_vae=True`, then `tiled_dit=True`.
- **Non-Tiled Output**: Fixed bug where output was undefined in non-tiled processing path.
- **Progress Bar**: Fixed display in ComfyUI using `cqdm` wrapper.
- **Full Mode VAE**: Added fallback mechanism to manually load VAE if model manager fails.

### ⚡ Performance

- **Deferred VAE Loading**: In `full` mode, VAE loading is deferred until strictly necessary.
- **90% VRAM Warning**: Added proactive warning when VRAM usage approaches 90%.
- **Memory Cleanup**: Added `torch.cuda.ipc_collect()` for better cleanup.

### 🛠 Refactoring

- **Progress Bar**: Rewrote to use single-line in-place updates (`\r`).
- **Default Settings**: Updated defaults for 16GB cards (`unload_dit=True`, tiled options enabled).
- **Documentation**: Expanded tooltips for all node parameters.

---

## [1.0.1] - 2025-12-06

### 🐛 Bug Fixes

- **Shape Mismatch**: Fixed error for small input frames with correct padding logic.
- **VRAM Cleanup**: VRAM immediately freed at start of processing to prevent OOM.

### 🚀 New Features

- **CPU Offload**: Added `keep_models_on_cpu` option to keep models in RAM.
- **FPS Reporting**: Added accurate FPS calculation and peak VRAM reporting.

### ⚡ Performance

- **Native PyTorch**: Replaced `einops` operations with native PyTorch ops where possible.
- **Conv3d Workaround**: Added memory optimization for Conv3d operations.

---

## [1.0.0] - 2025-10-21

### 🚀 Initial Release

- **Video Super Resolution**: ComfyUI node for FlashVSR video upscaling.
- **Three Modes**: `tiny`, `tiny-long`, and `full` processing modes.
- **Tiling Support**: Spatial tiling for VAE and DiT to reduce VRAM.
- **Sparse_Sage Attention**: Replaced Block-Sparse-Attention for RTX 50 series support.
- **Long Video Pipeline**: Streaming mode for very long videos with minimal VRAM spike.
