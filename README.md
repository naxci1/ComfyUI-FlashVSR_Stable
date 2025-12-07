# ComfyUI-FlashVSR_Ultra_Fast
Running FlashVSR on lower VRAM without any artifacts.

## Changelog
#### 06.12.2025
- **Bug Fix**: Fixed a shape mismatch error for small input frames by implementing correct padding logic.
- **Optimization**: VRAM is now immediately freed at the start of processing to prevent OOM errors.
- **New Feature**: Added `enable_debug` option for extensive logging (input shapes, tile stats, VRAM usage, processing time).
- **New Feature**: Added `keep_models_on_cpu` option to keep models in RAM (CPU) instead of VRAM, which is useful for GPUs with limited VRAM (e.g., 16GB).
- **Enhancement**: Added accurate FPS calculation and peak VRAM reporting (using `max_memory_reserved`) to the logs.
- **Optimization**: Replaced `einops` operations with native PyTorch ops for potential performance gains.
- **Optimization**: Added "Conv3d memory workaround" for improved compatibility with newer PyTorch versions.
- **Optimization**: Added `torch.cuda.ipc_collect()` for better memory cleanup.
- **Refactor**: Cleaned up code and improved error handling for imports.

#### 2025-10-24
- Added long video pipeline that significantly reduces VRAM usage when upscaling long videos.

#### 2025-10-22
- Replaced `Block-Sparse-Attention` with `Sparse_Sage`, removing the need to compile any custom kernels.  
- Added support for running on RTX 50 series GPUs.

#### 2025-10-21
- Initial release of this project, introducing features such as `tile_dit` to significantly reduce VRAM usage.

## Preview
![](./img/preview.jpg)

## Usage
- **mode:**  
`tiny` -> faster (default); `full` -> higher quality  
- **scale:**  
`4` is always better, unless you are low on VRAM then use `2`    
- **color_fix:**  
Use wavelet transform to correct the color of output video.  
- **tiled_vae:**  
Set to True for lower VRAM consumption during decoding at the cost of speed.  
- **tiled_dit:**  
Significantly reduces VRAM usage at the cost of speed.
- **tile\_size, tile\_overlap**:  
How to split the input video.  
- **unload_dit:**  
Unload DiT before decoding to reduce VRAM peak at the cost of speed.  
- **enable_debug:**
Enable extensive logging for debugging purposes. Displays detailed information about device status, input dimensions, and per-tile processing statistics.
- **keep_models_on_cpu:**
If enabled, models will be moved to RAM (CPU) after processing instead of remaining in VRAM. This helps prevent OOM errors on systems with limited VRAM.

## Installation

#### nodes: 

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Fast/requirements.txt
```
📢: For Turing or older GPUs, please install `triton<3.3.0`:

```bash
# Windows
python -m pip install -U triton-windows<3.3.0
# Linux
python -m pip install -U triton<3.3.0
```

#### models:

- Download the entire `FlashVSR` folder with all the files inside it from [here](https://huggingface.co/JunhaoZhuang/FlashVSR) and put it in the `ComfyUI/models` directory.

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## Acknowledgments
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
