"""
Example: Using FlashVSR GGUF Loader
====================================
This script demonstrates how to use the GGUF loader outside of ComfyUI
for testing and debugging purposes.
"""

import sys
import os
import torch

# Add the repository to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock ComfyUI modules (for standalone testing)
from unittest.mock import MagicMock
sys.modules['folder_paths'] = MagicMock()
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()


def test_tensor_utils():
    """Test the tensor utility functions."""
    print("\n" + "="*60)
    print("Testing Tensor Utils")
    print("="*60)
    
    from src.models.tensor_utils import (
        has_raw_shape_metadata,
        reshape_flattened_tensor,
        process_gguf_tensor,
        validate_5d_tensor_shape
    )
    
    # Test 1: Metadata detection
    print("\n1. Testing metadata detection...")
    metadata_5d = {'raw_shape': [1536, 16, 1, 2, 2]}
    metadata_4d = {'raw_shape': [64, 128]}
    metadata_none = {}
    
    assert has_raw_shape_metadata(metadata_5d) == True
    assert has_raw_shape_metadata(metadata_4d) == False
    assert has_raw_shape_metadata(metadata_none) == False
    print("   ✓ Metadata detection works correctly")
    
    # Test 2: Tensor reshaping
    print("\n2. Testing tensor reshaping...")
    flat_tensor = torch.randn(98304)  # 1536 * 16 * 1 * 2 * 2
    raw_shape = [1536, 16, 1, 2, 2]
    
    reshaped = reshape_flattened_tensor(flat_tensor, raw_shape, "test_tensor")
    assert reshaped.shape == torch.Size([1536, 16, 1, 2, 2])
    print(f"   ✓ Successfully reshaped from {flat_tensor.shape} to {reshaped.shape}")
    
    # Test 3: Shape validation
    print("\n3. Testing shape validation...")
    valid_shape = (1536, 16, 1, 2, 2)
    validate_5d_tensor_shape(valid_shape, "test")
    print(f"   ✓ Shape {valid_shape} validated successfully")
    
    # Test 4: Process GGUF tensor
    print("\n4. Testing GGUF tensor processing...")
    processed = process_gguf_tensor(flat_tensor, metadata_5d, "patch_embedding.weight")
    assert processed.shape == torch.Size([1536, 16, 1, 2, 2])
    print(f"   ✓ Processed tensor: {processed.shape}")
    
    print("\n" + "="*60)
    print("All tensor utils tests passed! ✓")
    print("="*60)


def test_state_dict_loading():
    """Test state_dict loading (requires mock GGUF file)."""
    print("\n" + "="*60)
    print("Testing State Dict Loading")
    print("="*60)
    
    try:
        import gguf
        print("✓ GGUF library is installed")
    except ImportError:
        print("✗ GGUF library not installed. Run: pip install gguf")
        return
    
    from src.models.utils import load_state_dict
    
    print("\n  Note: To test actual GGUF loading, you need a real GGUF file.")
    print("  Place a FlashVSR .gguf file in your test directory and update this script.")
    
    # Example usage (would work with a real file):
    # state_dict = load_state_dict("path/to/model.gguf", torch_dtype=torch.float16)
    # print(f"Loaded {len(state_dict)} tensors")
    
    print("\n" + "="*60)
    print("State dict loading test completed")
    print("="*60)


def example_workflow():
    """Example of how to use the GGUF loader in a workflow."""
    print("\n" + "="*60)
    print("Example Workflow")
    print("="*60)
    
    print("""
    # In your ComfyUI workflow:
    
    1. Add 'FlashVSR GGUF Loader' node
    2. Select your .gguf file from the dropdown
    3. Choose precision (float16 recommended)
    4. Connect output to FlashVSR pipeline nodes
    
    # Programmatically:
    
    from src.models.utils import load_state_dict
    from src.models.model_manager import ModelManager
    
    # Load GGUF model
    state_dict = load_state_dict("model.gguf", torch_dtype=torch.float16)
    
    # Create model manager
    model_manager = ModelManager(
        torch_dtype=torch.float16,
        device="cuda",
        file_path_list=["model.gguf"]
    )
    
    # Use with FlashVSR pipeline
    # pipeline = FlashVSRFullPipeline(model_manager=model_manager)
    # result = pipeline(prompt="...", ...)
    """)
    
    print("="*60)


def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("  FlashVSR GGUF Loader - Example & Test Suite")
    print("="*70)
    
    try:
        # Run tensor utils tests
        test_tensor_utils()
        
        # Run state dict loading tests
        test_state_dict_loading()
        
        # Show example workflow
        example_workflow()
        
        print("\n" + "="*70)
        print("  All tests completed successfully! ✓")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
