# FlashVSR GGUF Loader - Path Fix Summary

## Problem Fixed
The GGUF loader was incorrectly looking in the default 'checkpoints' folder instead of the FlashVSR-specific directory `models/FlashVSR-v1.1/`.

## Changes Made

### 1. Updated Path Registration
**Before:**
- Looked in `folder_paths.get_folder_paths("checkpoints")`
- Scanned all checkpoint directories

**After:**
- Looks in `folder_paths.models_dir + model_name` 
- Specifically scans `models/FlashVSR/` and `models/FlashVSR-v1.1/`
- Follows same pattern as nodes.py (line 711)

### 2. Added Model Name Selector
**New Input Field:**
```python
"model_name": (["FlashVSR", "FlashVSR-v1.1"], {
    "default": "FlashVSR-v1.1"
})
```

This allows users to select which FlashVSR model directory to use in the ComfyUI interface.

### 3. Updated File Discovery Logic
**File Scanning:**
- Iterates through both FlashVSR model directories
- Finds all `.gguf` files in: `models/FlashVSR/` and `models/FlashVSR-v1.1/`
- Prevents duplicates using a `seen_files` set

**Path Construction:**
```python
model_path = os.path.join(folder_paths.models_dir, model_name)
file_path = os.path.join(model_path, gguf_file)
```

### 4. Added Fallback Logic
If GGUF file is not found in the selected model directory, it checks the alternate directory:
- If selected `FlashVSR-v1.1` but file is in `FlashVSR`, it will find it
- Shows warning: "⚠️ Found GGUF file in {alt_model} instead of {model_name}"

### 5. Enhanced Logging
**New Log Messages:**
```
🚀 FlashVSR GGUF Engine Active: Loading {gguf_file}
📂 Model directory: {model_path}
📄 Loading GGUF file: {file_path}
```

**Error Messages:**
```
GGUF file not found: {gguf_file}
Expected location: {file_path}
Please place GGUF files in: {model_path}
```

### 6. Integration with 5D Reshaping
The existing 5D tensor reshaping logic is automatically triggered:
```python
state_dict = load_state_dict(file_path, torch_dtype=target_dtype)
```

This calls `load_state_dict_from_gguf()` which processes metadata and reshapes tensors.

## File Changes

### Modified: `flashvsr_gguf_node.py`
- Updated `INPUT_TYPES()` class method (lines 30-66)
  - Added `model_name` input field
  - Changed directory scanning from checkpoints to FlashVSR directories
  
- Updated `load_gguf_model()` method (lines 74-113)
  - Added `model_name` parameter
  - Changed path construction logic
  - Added fallback directory checking
  - Enhanced logging with emoji icons

### Created: `test_gguf_path_fix.py`
- 9 comprehensive tests for path handling
- Tests model_name input field
- Tests directory scanning
- Tests fallback logic
- Tests error handling

## Usage in ComfyUI

### Node Inputs
1. **model_name**: Select "FlashVSR" or "FlashVSR-v1.1" (default: FlashVSR-v1.1)
2. **gguf_file**: Select from available .gguf files found in the selected directory
3. **torch_dtype**: Choose precision (auto, float32, float16, bfloat16)

### Expected Directory Structure
```
ComfyUI/
└── models/
    ├── FlashVSR/
    │   ├── model.gguf
    │   └── diffusion_pytorch_model_streaming_dmd.safetensors
    └── FlashVSR-v1.1/
        ├── model.gguf
        └── diffusion_pytorch_model_streaming_dmd.safetensors
```

## Verification

### Syntax Check
✅ Python syntax validation passed

### Key Improvements
1. ✅ Correct path: Uses `models/FlashVSR-v1.1/` instead of checkpoints
2. ✅ Model selector: Added `model_name` UI input field
3. ✅ Integration: 5D reshaping logic automatically triggered
4. ✅ Logging: Added "🚀 FlashVSR GGUF Engine Active" message
5. ✅ Error handling: Clear messages showing expected file location
6. ✅ Fallback: Checks alternate directory if file not found

## Backward Compatibility
The changes maintain compatibility with the 5D tensor reshaping infrastructure:
- Still uses `load_state_dict()` which calls `load_state_dict_from_gguf()`
- Still processes metadata and reshapes tensors automatically
- Still logs 5D tensor reshaping: "5D tensor: {name} -> {shape}"
