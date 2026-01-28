# FlashVSR Init Pipeline - Complete Rewrite Summary

## Changes Made

### Problem Solved
The FlashVSR Init Pipeline node was using hardcoded paths and didn't support dynamic file selection. It couldn't load .gguf files and lacked a proper file selector UI.

### Solution Implemented

#### 1. Updated FlashVSRNodeInitPipe Class (nodes.py)

**New INPUT_TYPES:**
- **`model_version`**: Dropdown to select FlashVSR directory ("FlashVSR" or "FlashVSR-v1.1")
- **`model_file`**: Dynamic dropdown showing all .safetensors and .gguf files found in the selected directory
- Scans both FlashVSR directories and aggregates all model files
- Shows actual filenames (e.g., "model_f16.gguf") instead of just version names

**Updated main() Method:**
- Now accepts `model_version` and `model_file` parameters
- Passes selected file to `init_pipeline` function
- Maintains all existing functionality (VAE selection, precision, etc.)

#### 2. Updated init_pipeline Function (nodes.py)

**New Signature:**
```python
def init_pipeline(model, mode, device, dtype, vae_model="Wan2.1", model_file=None)
```

**Key Changes:**
- Added `model_file` parameter with default None
- Falls back to "diffusion_pytorch_model_streaming_dmd.safetensors" if model_file is None
- Constructs dynamic path: `model_path / model_file`
- Detects file type (.gguf vs .safetensors) for logging
- Removed hardcoded "diffusion_pytorch_model_streaming_dmd.safetensors" requirement
- Enhanced logging with emoji icons showing file type

**File Type Support:**
- **.safetensors**: Uses existing ModelManager loading logic
- **.gguf**: Uses same loading path, which triggers 5D tensor reshaping via load_state_dict_from_gguf()

#### 3. What Users Will See

**In ComfyUI Node:**
```
FlashVSR Init Pipeline
├── model_version: FlashVSR-v1.1 [dropdown]
├── model_file: model_f16.gguf [dropdown with all files]
├── mode: tiny [tiny/tiny-long/full]
├── vae_model: Wan2.1 [5 VAE options]
├── force_offload: True
├── precision: auto [fp16/bf16/auto]
├── device: auto [auto/cuda/cpu]
└── attention_mode: sparse_sage_attention
```

**Model File Dropdown Shows:**
- All .safetensors files in models/FlashVSR/ and models/FlashVSR-v1.1/
- All .gguf files in models/FlashVSR/ and models/FlashVSR-v1.1/
- Example: "model_f16.gguf", "diffusion_pytorch_model_streaming_dmd.safetensors"

**In Console Logs:**
```
🚀 Loading Model File: model_f16.gguf (GGUF with 5D reshaping)
```
or
```
🚀 Loading Model File: diffusion_pytorch_model_streaming_dmd.safetensors (SafeTensors)
```

### Integration with Existing Code

#### GGUF Support
The ModelManager.load_models() function automatically uses load_state_dict() which:
1. Detects .gguf extension
2. Calls load_state_dict_from_gguf()
3. Applies 5D tensor reshaping from tensor_utils.py
4. Returns reshaped tensors ready for the pipeline

This means GGUF files "just work" without additional changes because the infrastructure was already built in the previous commits.

#### Backward Compatibility
- If no model_file is specified, defaults to standard .safetensors file
- Existing workflows will continue to work
- Node category and return types unchanged

### Testing

Created comprehensive test suite (`test_init_pipe_dynamic_selector.py`):
- ✅ 9 tests all passing
- Tests file scanning logic
- Tests both file type support (.safetensors and .gguf)
- Tests parameter structure
- Tests path construction
- Tests default fallback behavior

### File Changes

**Modified:**
- `nodes.py` (2 functions updated)
  - FlashVSRNodeInitPipe class completely rewritten
  - init_pipeline function updated with model_file support

**Created:**
- `test_init_pipe_dynamic_selector.py` (9 comprehensive tests)
- `INIT_PIPELINE_REWRITE_SUMMARY.md` (this file)

### Key Features

1. ✅ **Dynamic File Selector**: Shows all .safetensors and .gguf files
2. ✅ **Unified Loader Logic**: Automatically detects and handles both file types
3. ✅ **No Hardcoded Paths**: Uses user-selected file from dropdown
4. ✅ **5D Tensor Reshaping**: GGUF files automatically use tensor_utils.py reshaping
5. ✅ **Enhanced Logging**: Shows which file type is being loaded
6. ✅ **Backward Compatible**: Defaults to standard file if none selected

### How It Works

1. **On Node Load:**
   - ComfyUI calls INPUT_TYPES()
   - Scans models/FlashVSR/ and models/FlashVSR-v1.1/
   - Finds all .safetensors and .gguf files
   - Populates dropdown with found files

2. **When User Selects File:**
   - User picks model_version (directory)
   - User picks model_file from dropdown
   - Node passes both to main() method

3. **During Initialization:**
   - main() calls init_pipeline(model_version, ..., model_file=selected_file)
   - init_pipeline constructs full path
   - Passes to ModelManager.load_models([ckpt_path])
   - ModelManager detects file type and loads appropriately

4. **For GGUF Files:**
   - load_state_dict() detects .gguf extension
   - Calls load_state_dict_from_gguf()
   - process_gguf_tensor() reshapes 5D tensors
   - Returns ready-to-use state_dict

### Example Usage

**Loading .gguf file:**
```python
# User selects in UI:
# - model_version: "FlashVSR-v1.1"
# - model_file: "flashvsr_model_f16.gguf"

# Results in:
# File path: models/FlashVSR-v1.1/flashvsr_model_f16.gguf
# Loading: GGUF with automatic 5D tensor reshaping
# Console: "🚀 Loading Model File: flashvsr_model_f16.gguf (GGUF with 5D reshaping)"
```

**Loading .safetensors file:**
```python
# User selects in UI:
# - model_version: "FlashVSR-v1.1"
# - model_file: "diffusion_pytorch_model_streaming_dmd.safetensors"

# Results in:
# File path: models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors
# Loading: Standard safetensors
# Console: "🚀 Loading Model File: diffusion_pytorch_model_streaming_dmd.safetensors (SafeTensors)"
```

### Verification

✅ Python syntax validation passed
✅ All 9 unit tests passed
✅ File scanning logic tested
✅ Both file types supported
✅ Path construction verified
✅ Default fallback tested
✅ Integration with ModelManager confirmed

### Result

Users can now select any .gguf or .safetensors file from the FlashVSR model directories directly in the ComfyUI interface. The node automatically handles both file types, with GGUF files triggering the 5D tensor reshaping logic we built previously.

No more hardcoded paths! 🎉
