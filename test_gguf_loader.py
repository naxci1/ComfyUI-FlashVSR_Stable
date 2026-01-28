"""
Tests for GGUF Loader Functionality
====================================
Tests for loading FlashVSR models from GGUF format with 5D tensor support.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch, mock_open
import tempfile
import numpy as np

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock ComfyUI modules before importing our code
sys.modules['folder_paths'] = MagicMock()
sys.modules['folder_paths'].get_folder_paths = MagicMock(return_value=["/tmp/models"])
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()

import torch
from src.models.tensor_utils import (
    has_raw_shape_metadata,
    reshape_flattened_tensor,
    process_gguf_tensor,
    validate_5d_tensor_shape
)


class TestTensorUtils(unittest.TestCase):
    """Test tensor utility functions for GGUF loading."""
    
    def test_has_raw_shape_metadata_valid_5d(self):
        """Test detection of valid 5D raw_shape metadata."""
        metadata = {'raw_shape': [1536, 16, 1, 2, 2]}
        self.assertTrue(has_raw_shape_metadata(metadata))
    
    def test_has_raw_shape_metadata_no_metadata(self):
        """Test with missing metadata."""
        metadata = {}
        self.assertFalse(has_raw_shape_metadata(metadata))
    
    def test_has_raw_shape_metadata_wrong_dimensions(self):
        """Test with non-5D shape."""
        metadata = {'raw_shape': [1536, 16]}
        self.assertFalse(has_raw_shape_metadata(metadata))
    
    def test_has_raw_shape_metadata_invalid_type(self):
        """Test with invalid metadata type."""
        metadata = "not a dict"
        self.assertFalse(has_raw_shape_metadata(metadata))
    
    def test_reshape_flattened_tensor_correct_size(self):
        """Test reshaping a flattened tensor to 5D with correct size."""
        # Create a flattened tensor: 1536 * 16 * 1 * 2 * 2 = 98304 elements
        flat_tensor = torch.randn(98304)
        raw_shape = [1536, 16, 1, 2, 2]
        
        reshaped = reshape_flattened_tensor(flat_tensor, raw_shape, "test_tensor")
        
        self.assertEqual(reshaped.shape, torch.Size([1536, 16, 1, 2, 2]))
        self.assertEqual(reshaped.numel(), 98304)
    
    def test_reshape_flattened_tensor_2d_input(self):
        """Test reshaping a 2D tensor to 5D."""
        # Create a 2D tensor that can be reshaped to [8, 4, 2, 2, 2]
        tensor_2d = torch.randn(32, 8)  # 256 elements total
        raw_shape = [8, 4, 2, 2, 2]
        
        reshaped = reshape_flattened_tensor(tensor_2d, raw_shape, "test_2d")
        
        self.assertEqual(reshaped.shape, torch.Size([8, 4, 2, 2, 2]))
        self.assertEqual(reshaped.numel(), 256)
    
    def test_reshape_flattened_tensor_size_mismatch(self):
        """Test error handling for size mismatch."""
        flat_tensor = torch.randn(100)  # Wrong size
        raw_shape = [1536, 16, 1, 2, 2]  # Expects 98304 elements
        
        with self.assertRaises(ValueError) as context:
            reshape_flattened_tensor(flat_tensor, raw_shape, "bad_tensor")
        
        self.assertIn("size mismatch", str(context.exception).lower())
    
    def test_reshape_flattened_tensor_invalid_shape_type(self):
        """Test error handling for invalid shape type."""
        flat_tensor = torch.randn(100)
        raw_shape = "not a list"
        
        with self.assertRaises(ValueError) as context:
            reshape_flattened_tensor(flat_tensor, raw_shape, "bad_shape")
        
        self.assertIn("must be a list or tuple", str(context.exception))
    
    def test_process_gguf_tensor_with_5d_metadata(self):
        """Test processing a tensor with 5D metadata."""
        # Create flattened tensor
        flat_tensor = torch.randn(98304)
        metadata = {'raw_shape': [1536, 16, 1, 2, 2]}
        
        processed = process_gguf_tensor(flat_tensor, metadata, "patch_embedding.weight")
        
        self.assertEqual(processed.shape, torch.Size([1536, 16, 1, 2, 2]))
    
    def test_process_gguf_tensor_without_metadata(self):
        """Test processing a tensor without 5D metadata (should pass through)."""
        tensor = torch.randn(64, 128)
        metadata = {}
        
        processed = process_gguf_tensor(tensor, metadata, "normal_weight")
        
        # Should be unchanged
        self.assertEqual(processed.shape, torch.Size([64, 128]))
        self.assertTrue(torch.equal(processed, tensor))
    
    def test_process_gguf_tensor_with_4d_metadata(self):
        """Test processing with non-5D metadata (should pass through)."""
        tensor = torch.randn(64, 128)
        metadata = {'raw_shape': [64, 128]}  # 2D, not 5D
        
        processed = process_gguf_tensor(tensor, metadata, "normal_weight")
        
        # Should be unchanged since raw_shape is not 5D
        self.assertEqual(processed.shape, torch.Size([64, 128]))
    
    def test_validate_5d_tensor_shape_valid(self):
        """Test validation of valid 5D shape."""
        valid_shape = (1536, 16, 1, 2, 2)
        self.assertTrue(validate_5d_tensor_shape(valid_shape, "test_tensor"))
    
    def test_validate_5d_tensor_shape_wrong_dimensions(self):
        """Test validation fails for non-5D shape."""
        invalid_shape = (64, 128)
        
        with self.assertRaises(ValueError) as context:
            validate_5d_tensor_shape(invalid_shape, "bad_tensor")
        
        self.assertIn("Expected 5D tensor", str(context.exception))
    
    def test_validate_5d_tensor_shape_negative_dim(self):
        """Test validation fails for negative dimension."""
        invalid_shape = (1536, -16, 1, 2, 2)
        
        with self.assertRaises(ValueError) as context:
            validate_5d_tensor_shape(invalid_shape, "bad_tensor")
        
        self.assertIn("must be positive", str(context.exception))
    
    def test_validate_5d_tensor_shape_too_large_channels(self):
        """Test validation fails for unreasonably large channel dimensions."""
        invalid_shape = (200000, 16, 1, 2, 2)  # > 100000
        
        with self.assertRaises(ValueError) as context:
            validate_5d_tensor_shape(invalid_shape, "bad_tensor")
        
        self.assertIn("too large", str(context.exception).lower())
    
    def test_validate_5d_tensor_shape_too_large_spatial(self):
        """Test validation fails for unreasonably large spatial dimensions."""
        invalid_shape = (64, 32, 5000, 2, 2)  # spatial > 1000
        
        with self.assertRaises(ValueError) as context:
            validate_5d_tensor_shape(invalid_shape, "bad_tensor")
        
        self.assertIn("too large", str(context.exception).lower())


class TestGGUFLoading(unittest.TestCase):
    """Test GGUF file loading functionality."""
    
    @patch('src.models.utils.gguf')
    def test_load_state_dict_from_gguf_basic(self, mock_gguf_module):
        """Test basic GGUF loading without 5D tensors."""
        from src.models.utils import load_state_dict_from_gguf
        
        # Create mock GGUF reader
        mock_reader = MagicMock()
        mock_gguf_module.GGUFReader.return_value = mock_reader
        
        # Create mock tensor data (2D weight)
        mock_tensor = MagicMock()
        mock_tensor.name = "layer.weight"
        mock_tensor.data = np.random.randn(64, 128).astype(np.float32)
        
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}
        
        # Test loading
        state_dict = load_state_dict_from_gguf("/tmp/test.gguf", torch_dtype=torch.float32)
        
        # Verify
        self.assertIn("layer.weight", state_dict)
        self.assertEqual(state_dict["layer.weight"].shape, torch.Size([64, 128]))
    
    @patch('src.models.utils.gguf')
    def test_load_state_dict_from_gguf_with_5d_tensor(self, mock_gguf_module):
        """Test GGUF loading with 5D tensor metadata."""
        from src.models.utils import load_state_dict_from_gguf
        
        # Create mock GGUF reader
        mock_reader = MagicMock()
        mock_gguf_module.GGUFReader.return_value = mock_reader
        
        # Create mock flattened 5D tensor (1536 * 16 * 1 * 2 * 2 = 98304)
        mock_tensor = MagicMock()
        mock_tensor.name = "patch_embedding.weight"
        mock_tensor.data = np.random.randn(98304).astype(np.float16)
        mock_tensor.metadata = {'raw_shape': [1536, 16, 1, 2, 2]}
        
        mock_reader.tensors = [mock_tensor]
        mock_reader.fields = {}
        
        # Test loading
        state_dict = load_state_dict_from_gguf("/tmp/test.gguf", torch_dtype=torch.float16)
        
        # Verify 5D tensor was reshaped correctly
        self.assertIn("patch_embedding.weight", state_dict)
        self.assertEqual(
            state_dict["patch_embedding.weight"].shape, 
            torch.Size([1536, 16, 1, 2, 2])
        )
    
    def test_load_state_dict_handles_gguf_extension(self):
        """Test that load_state_dict routes .gguf files correctly."""
        from src.models.utils import load_state_dict
        
        with patch('src.models.utils.load_state_dict_from_gguf') as mock_gguf_loader:
            mock_gguf_loader.return_value = {"test": torch.randn(10)}
            
            result = load_state_dict("/tmp/model.gguf")
            
            # Verify GGUF loader was called
            mock_gguf_loader.assert_called_once_with("/tmp/model.gguf", torch_dtype=None)


class TestGGUFNode(unittest.TestCase):
    """Test the FlashVSR GGUF Loader ComfyUI node."""
    
    def test_node_imports(self):
        """Test that the GGUF node can be imported."""
        try:
            from flashvsr_gguf_node import FlashVSR_GGUF_Loader
            self.assertTrue(hasattr(FlashVSR_GGUF_Loader, 'INPUT_TYPES'))
            self.assertTrue(hasattr(FlashVSR_GGUF_Loader, 'load_gguf_model'))
        except ImportError:
            # If gguf library is not installed, this is expected
            self.skipTest("GGUF library not installed")
    
    def test_node_input_types(self):
        """Test that the node has correct input types defined."""
        try:
            from flashvsr_gguf_node import FlashVSR_GGUF_Loader
            
            input_types = FlashVSR_GGUF_Loader.INPUT_TYPES()
            
            self.assertIn('required', input_types)
            self.assertIn('gguf_file', input_types['required'])
            self.assertIn('torch_dtype', input_types['required'])
            
            # Check dtype options
            dtype_options = input_types['required']['torch_dtype'][0]
            self.assertIn('auto', dtype_options)
            self.assertIn('float16', dtype_options)
            self.assertIn('float32', dtype_options)
            self.assertIn('bfloat16', dtype_options)
        except ImportError:
            self.skipTest("GGUF library not installed")
    
    def test_node_return_types(self):
        """Test that the node has correct return types."""
        try:
            from flashvsr_gguf_node import FlashVSR_GGUF_Loader
            
            self.assertEqual(FlashVSR_GGUF_Loader.RETURN_TYPES, ("MODEL_MANAGER",))
            self.assertEqual(FlashVSR_GGUF_Loader.FUNCTION, "load_gguf_model")
            self.assertEqual(FlashVSR_GGUF_Loader.CATEGORY, "FlashVSR")
        except ImportError:
            self.skipTest("GGUF library not installed")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete GGUF loading pipeline."""
    
    def test_tensor_roundtrip(self):
        """Test that a tensor can be flattened and reshaped correctly."""
        # Create original 5D tensor
        original = torch.randn(8, 4, 2, 2, 2)
        original_shape = list(original.shape)
        
        # Flatten it
        flattened = original.flatten()
        
        # Reshape back using our function
        reshaped = reshape_flattened_tensor(flattened, original_shape, "test")
        
        # Should be identical
        self.assertEqual(reshaped.shape, original.shape)
        self.assertTrue(torch.allclose(reshaped, original))
    
    def test_process_tensor_with_different_dtypes(self):
        """Test processing tensors with different data types."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                tensor = torch.randn(98304, dtype=dtype)
                metadata = {'raw_shape': [1536, 16, 1, 2, 2]}
                
                processed = process_gguf_tensor(tensor, metadata, "test")
                
                self.assertEqual(processed.dtype, dtype)
                self.assertEqual(processed.shape, torch.Size([1536, 16, 1, 2, 2]))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
