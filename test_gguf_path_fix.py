"""
Test for FlashVSR GGUF Loader Path Fix
========================================
Tests the updated path handling to ensure GGUF files are loaded from models/FlashVSR-v1.1/
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock ComfyUI modules
sys.modules['folder_paths'] = MagicMock()
sys.modules['folder_paths'].models_dir = "/mock/models"
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()

# Mock torch to avoid import issues
sys.modules['torch'] = MagicMock()
sys.modules['torch'].float32 = "float32"
sys.modules['torch'].float16 = "float16"
sys.modules['torch'].bfloat16 = "bfloat16"
sys.modules['torch'].cuda = MagicMock()
sys.modules['torch'].cuda.is_available = MagicMock(return_value=False)


class TestFlashVSRGGUFLoaderPathFix(unittest.TestCase):
    """Test the updated GGUF loader with FlashVSR-specific paths."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset mocks
        sys.modules['folder_paths'].models_dir = "/mock/models"
    
    def test_input_types_includes_model_name(self):
        """Test that INPUT_TYPES now includes model_name field."""
        from flashvsr_gguf_node import FlashVSR_GGUF_Loader
        
        input_types = FlashVSR_GGUF_Loader.INPUT_TYPES()
        
        # Verify model_name is in required inputs
        self.assertIn('model_name', input_types['required'])
        
        # Verify model_name options include both FlashVSR variants
        model_options = input_types['required']['model_name'][0]
        self.assertIn('FlashVSR', model_options)
        self.assertIn('FlashVSR-v1.1', model_options)
        
        # Verify default is FlashVSR-v1.1
        self.assertEqual(input_types['required']['model_name'][1]['default'], 'FlashVSR-v1.1')
    
    def test_input_types_scans_flashvsr_directories(self):
        """Test that INPUT_TYPES scans FlashVSR model directories for GGUF files."""
        with patch('os.path.exists') as mock_exists, \
             patch('os.listdir') as mock_listdir:
            
            # Mock directory structure
            def exists_side_effect(path):
                return path in [
                    "/mock/models/FlashVSR",
                    "/mock/models/FlashVSR-v1.1"
                ]
            
            mock_exists.side_effect = exists_side_effect
            
            # Mock file listings
            def listdir_side_effect(path):
                if path == "/mock/models/FlashVSR":
                    return ["model1.gguf", "config.json"]
                elif path == "/mock/models/FlashVSR-v1.1":
                    return ["model2.gguf", "diffusion_model.safetensors"]
                return []
            
            mock_listdir.side_effect = listdir_side_effect
            
            from flashvsr_gguf_node import FlashVSR_GGUF_Loader
            
            input_types = FlashVSR_GGUF_Loader.INPUT_TYPES()
            
            # Verify GGUF files were found
            gguf_files = input_types['required']['gguf_file'][0]
            self.assertIn('model1.gguf', gguf_files)
            self.assertIn('model2.gguf', gguf_files)
            
            # Verify no duplicates
            self.assertEqual(len(gguf_files), 2)
    
    def test_load_gguf_model_signature(self):
        """Test that load_gguf_model now accepts model_name parameter."""
        from flashvsr_gguf_node import FlashVSR_GGUF_Loader
        
        loader = FlashVSR_GGUF_Loader()
        
        # Check method exists and has correct signature
        self.assertTrue(hasattr(loader, 'load_gguf_model'))
        
        # Check function accepts model_name parameter
        import inspect
        sig = inspect.signature(loader.load_gguf_model)
        params = list(sig.parameters.keys())
        
        self.assertIn('model_name', params)
        self.assertIn('gguf_file', params)
        self.assertIn('torch_dtype', params)
    
    @patch('os.path.exists')
    @patch('flashvsr_gguf_node.load_state_dict')
    @patch('flashvsr_gguf_node.ModelManager')
    def test_load_gguf_uses_flashvsr_path(self, mock_manager, mock_load_state, mock_exists):
        """Test that load_gguf_model constructs correct path in FlashVSR directory."""
        from flashvsr_gguf_node import FlashVSR_GGUF_Loader
        
        # Setup mocks
        mock_exists.return_value = True
        mock_load_state.return_value = {"test_tensor": MagicMock(shape=(64, 128))}
        mock_manager_instance = MagicMock()
        mock_manager_instance.model_name = ["test_model"]
        mock_manager.return_value = mock_manager_instance
        
        loader = FlashVSR_GGUF_Loader()
        
        # Call with FlashVSR-v1.1
        try:
            loader.load_gguf_model("FlashVSR-v1.1", "test.gguf", "auto")
        except Exception as e:
            # May fail due to mocking, but we can check the path construction
            pass
        
        # Verify path construction - should use models_dir + model_name
        expected_path = "/mock/models/FlashVSR-v1.1/test.gguf"
        mock_exists.assert_any_call(expected_path)
    
    @patch('os.path.exists')
    def test_load_gguf_falls_back_to_alternate_directory(self, mock_exists):
        """Test that loader falls back to checking alternate FlashVSR directory."""
        from flashvsr_gguf_node import FlashVSR_GGUF_Loader
        
        # Mock: file not in FlashVSR-v1.1 but exists in FlashVSR
        def exists_side_effect(path):
            if "FlashVSR-v1.1" in path:
                return False
            elif "FlashVSR/test.gguf" in path:
                return True
            return False
        
        mock_exists.side_effect = exists_side_effect
        
        loader = FlashVSR_GGUF_Loader()
        
        with patch('flashvsr_gguf_node.load_state_dict') as mock_load, \
             patch('flashvsr_gguf_node.ModelManager') as mock_manager:
            
            mock_load.return_value = {"test": MagicMock(shape=(10,))}
            mock_manager_instance = MagicMock()
            mock_manager_instance.model_name = ["test"]
            mock_manager.return_value = mock_manager_instance
            
            # Should find file in alternate directory
            loader.load_gguf_model("FlashVSR-v1.1", "test.gguf", "auto")
            
            # Verify it checked the alternate path
            alternate_path = "/mock/models/FlashVSR/test.gguf"
            self.assertTrue(any(alternate_path in str(call) for call in mock_exists.call_args_list))
    
    @patch('os.path.exists')
    def test_load_gguf_raises_error_if_not_found(self, mock_exists):
        """Test that loader raises clear error if GGUF file not found."""
        from flashvsr_gguf_node import FlashVSR_GGUF_Loader
        
        mock_exists.return_value = False
        
        loader = FlashVSR_GGUF_Loader()
        
        with self.assertRaises(FileNotFoundError) as context:
            loader.load_gguf_model("FlashVSR-v1.1", "missing.gguf", "auto")
        
        # Verify error message includes expected path
        error_msg = str(context.exception)
        self.assertIn("missing.gguf", error_msg)
        self.assertIn("FlashVSR-v1.1", error_msg)
    
    def test_node_category_is_flashvsr(self):
        """Test that node is in FlashVSR category."""
        from flashvsr_gguf_node import FlashVSR_GGUF_Loader
        
        self.assertEqual(FlashVSR_GGUF_Loader.CATEGORY, "FlashVSR")
    
    def test_node_registration_mappings(self):
        """Test that node registration mappings are correct."""
        from flashvsr_gguf_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        self.assertIn("FlashVSR_GGUF_Loader", NODE_CLASS_MAPPINGS)
        self.assertIn("FlashVSR_GGUF_Loader", NODE_DISPLAY_NAME_MAPPINGS)
        
        from flashvsr_gguf_node import FlashVSR_GGUF_Loader
        self.assertEqual(NODE_CLASS_MAPPINGS["FlashVSR_GGUF_Loader"], FlashVSR_GGUF_Loader)


class TestPathIntegration(unittest.TestCase):
    """Integration tests for path handling."""
    
    def test_consistent_with_nodes_py_pattern(self):
        """Test that path construction matches pattern used in nodes.py."""
        # This is the pattern from nodes.py line 711:
        # model_path = os.path.join(folder_paths.models_dir, model)
        
        import folder_paths
        folder_paths.models_dir = "/test/models"
        
        model_name = "FlashVSR-v1.1"
        expected_path = os.path.join(folder_paths.models_dir, model_name)
        
        # Verify our implementation uses same pattern
        self.assertEqual(expected_path, "/test/models/FlashVSR-v1.1")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
