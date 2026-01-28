# FlashVSR GGUF Loader

## Overview

The FlashVSR GGUF Loader is a custom ComfyUI node that enables loading FlashVSR models in GGUF (GGML Unified Format) with full support for 5D tensors.

## Background: The 5D Tensor Challenge

### The Problem

The GGUF format specification has a limitation: **it only supports tensors up to 4 dimensions**. However, FlashVSR's video super-resolution architecture uses 3D convolution layers that require 5D tensors with shape `[out_channels, in_channels, temporal_depth, height, width]`.

Key tensor example:
- **Tensor**: `patch_embedding.weight`
- **Original Shape**: `[1536, 16, 1, 2, 2]` (5D)
- **Total Elements**: 98,304

### The Solution: "DNA" Encoding

To work around GGUF's 4D limitation, we implemented a "DNA" encoding strategy during conversion:

1. **Flattening**: The 5D tensor is flattened into a 1D array (98,304 elements)
2. **Metadata Storage**: The original 5D shape is preserved as `raw_shape` metadata in the GGUF file
3. **Automatic Reshaping**: During loading, the tensor is automatically reshaped back to 5D

## Features

### 🔍 Automatic 5D Detection
- Identifies tensors with `raw_shape` metadata indicating original 5D dimensions
- No manual intervention required

### ♻️ Dynamic Reshaping
- Automatically reshapes flattened tensors back to their original 5D dimensions
- Validates tensor sizes match expected dimensions

### 🛡️ Error Handling
- Comprehensive validation to prevent crashes
- Checks for size mismatches between flattened data and target shape
- Validates reasonable bounds for tensor dimensions

### 🎯 ComfyUI Integration
- Seamless integration with existing FlashVSR nodes
- Compatible with ModelManager infrastructure
- Supports multiple precision modes (FP32, FP16, BF16)

## Installation

1. **Install the GGUF library**:
```bash
pip install gguf
```

2. **Place GGUF files** in your ComfyUI checkpoints folder:
```
ComfyUI/models/checkpoints/your_model.gguf
```

## Usage

### In ComfyUI

1. Add the **FlashVSR GGUF Loader** node to your workflow
2. Select your GGUF file from the dropdown
3. Choose precision mode:
   - `auto` - Automatically selects FP16 (recommended)
   - `float32` - Full precision (higher memory)
   - `float16` - Half precision (balanced)
   - `bfloat16` - BFloat16 (best for RTX 3000+)
4. Connect the output to FlashVSR pipeline nodes

### Node Outputs

- **model_manager** - ModelManager instance containing the loaded model

## Technical Details

### Architecture

```
FlashVSR GGUF Loader
├── tensor_utils.py          # Core reshaping logic
│   ├── has_raw_shape_metadata()    # Detect 5D metadata
│   ├── reshape_flattened_tensor()  # Reshape operation
│   ├── process_gguf_tensor()       # Main processing
│   └── validate_5d_tensor_shape()  # Shape validation
│
├── utils.py (updated)       # File loading
│   └── load_state_dict_from_gguf() # GGUF reader
│
└── flashvsr_gguf_node.py   # ComfyUI node
    └── FlashVSR_GGUF_Loader       # Node class
```

### Metadata Format

The GGUF file stores 5D shape information in metadata:

```python
# Metadata stored in GGUF
{
    "patch_embedding.weight": {
        "raw_shape": [1536, 16, 1, 2, 2],
        # ... other metadata
    }
}
```

### Reshaping Process

1. **Load**: Read flattened tensor from GGUF (shape: `[98304]`)
2. **Detect**: Check for `raw_shape` metadata
3. **Validate**: Ensure `98304 == 1536 × 16 × 1 × 2 × 2`
4. **Reshape**: Transform to `[1536, 16, 1, 2, 2]`
5. **Convert**: Apply target dtype (FP16/FP32/BF16)

## Example: Converting Your Own Model

If you're converting a FlashVSR model to GGUF format, ensure you:

1. **Flatten 5D tensors** to 1D before writing to GGUF
2. **Store original shape** in metadata with key `raw_shape`
3. **Use FP16** to reduce file size (5.26GB → 2.64GB)

### Python Conversion Example

```python
import torch
import gguf

# Load your FlashVSR model
state_dict = torch.load("flashvsr.pth")

# Create GGUF writer
writer = gguf.GGUFWriter("flashvsr.gguf", "flashvsr")

for name, tensor in state_dict.items():
    if len(tensor.shape) == 5:
        # Store original shape in metadata
        original_shape = list(tensor.shape)
        
        # Flatten to 1D
        flat_tensor = tensor.flatten()
        
        # Write with metadata
        writer.add_tensor(name, flat_tensor.numpy(), 
                         metadata={'raw_shape': original_shape})
    else:
        # Normal tensors (≤4D) can be written directly
        writer.add_tensor(name, tensor.numpy())

writer.write_header_to_file()
writer.write_tensors_to_file()
writer.close()
```

## Model Compatibility

### Supported Models
- FlashVSR (Full, Tiny, Tiny-Long variants)
- DMD (Distribution Matching Distillation) based models
- Any video diffusion model using 3D convolutions with 5D tensors

### Supported Formats
- **Input**: `.gguf` files (with optional 5D tensor metadata)
- **Output**: PyTorch state_dict compatible with FlashVSR architecture

### Precision Modes
| Mode | Memory | Speed | Quality | Notes |
|------|--------|-------|---------|-------|
| FP32 | 2x | Slow | Best | For validation only |
| FP16 | 1x | Fast | Good | Recommended default |
| BF16 | 1x | Fastest | Best | RTX 3000+ GPUs |

## Troubleshooting

### "GGUF file not found"
- Ensure file is in `ComfyUI/models/checkpoints/`
- Check filename matches exactly (case-sensitive)

### "Size mismatch" error
- GGUF file may be corrupted
- Metadata `raw_shape` doesn't match actual tensor size
- Try reconverting the model

### "gguf library required"
- Install: `pip install gguf`
- Restart ComfyUI after installation

### "Tensor dimensions too large"
- Shape validation failed (safety check)
- Verify model is actually a FlashVSR model
- Check for corruption during conversion

## Performance

### Memory Usage
- **FP16 GGUF**: ~2.64 GB (from 5.26GB FP32)
- **Loading Time**: ~5-10 seconds (depends on disk speed)
- **VRAM Impact**: Same as loading from safetensors/pth

### Recommendations
- Use FP16 for most cases (good balance)
- Use BF16 for RTX 3000+ (best performance)
- Use FP32 only for debugging

## Testing

Run the test suite:
```bash
python test_gguf_loader.py
```

Tests cover:
- Tensor reshaping logic
- Metadata detection
- Error handling
- Integration with ModelManager

## Advanced: Custom GGUF Integration

The loader can be extended for other model architectures:

```python
from src.models.tensor_utils import process_gguf_tensor

# In your custom loader
for tensor in gguf_reader.tensors:
    # Automatically handles 5D tensors
    processed = process_gguf_tensor(
        tensor.data,
        tensor.metadata,
        tensor.name
    )
    state_dict[tensor.name] = processed
```

## Contributing

When working with GGUF support:
1. Maintain backward compatibility with safetensors/pth
2. Add tests for new tensor shapes
3. Document any new metadata formats
4. Validate with actual FlashVSR models

## License

Same as ComfyUI-FlashVSR_Stable (MIT License)

## Credits

- **FlashVSR**: Original architecture by JunhaoZhuang
- **GGUF Format**: GGML Unified Format by ggerganov
- **Implementation**: ComfyUI-FlashVSR_Stable community

## See Also

- [Main README](../README.md)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [FlashVSR Paper](https://arxiv.org/abs/2404.12242)
