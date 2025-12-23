
import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock ComfyUI modules
sys.modules['folder_paths'] = MagicMock()
sys.modules['folder_paths'].get_filename_list = MagicMock(return_value=[])
sys.modules['folder_paths'].models_dir = "/tmp/models"
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.utils'].ProgressBar = MagicMock()

# Mock dependencies that might be missing or require heavy setup
sys.modules['sageattention'] = MagicMock()
sys.modules['flash_attn'] = MagicMock()

import torch
# We need to ensure torch.cuda.is_available is mocked if no GPU
if not torch.cuda.is_available():
    torch.cuda.is_available = MagicMock(return_value=False)

from nodes import flashvsr, FlashVSRNodeInitPipe, FlashVSRNode, FlashVSRNodeAdv, VAE_MODEL_OPTIONS, VAE_MODEL_MAP
from nodes import estimate_vram_usage, get_optimal_settings, check_resources
from src.pipelines.flashvsr_full import FlashVSRFullPipeline
from src.models.wan_video_vae import WanVideoVAE, Wan22VideoVAE, LightX2VVAE, create_video_vae

class TestFlashVSRNodes(unittest.TestCase):
    def test_pipeline_instantiation(self):
        # We can't easily instantiate the full pipeline without models,
        # but we can check if the class loads and methods exist.
        self.assertTrue(hasattr(FlashVSRFullPipeline, '__call__'))

    def test_nodes_import(self):
        self.assertTrue(hasattr(FlashVSRNode, 'INPUT_TYPES'))
        self.assertTrue(hasattr(FlashVSRNodeAdv, 'INPUT_TYPES'))
        self.assertTrue(hasattr(FlashVSRNodeInitPipe, 'INPUT_TYPES'))

    def test_vae_model_options_available(self):
        """Test that VAE_MODEL_OPTIONS is properly defined and contains all 5 expected values."""
        self.assertIn("Wan2.1", VAE_MODEL_OPTIONS)
        self.assertIn("Wan2.2", VAE_MODEL_OPTIONS)
        self.assertIn("LightVAE_W2.1", VAE_MODEL_OPTIONS)
        self.assertIn("TAE_W2.2", VAE_MODEL_OPTIONS)
        self.assertIn("LightTAE_HY1.5", VAE_MODEL_OPTIONS)
        self.assertEqual(len(VAE_MODEL_OPTIONS), 5)
    
    def test_vae_model_map_configured(self):
        """Test that VAE_MODEL_MAP is correctly configured with DISTINCT files and URLs."""
        self.assertIn("Wan2.1", VAE_MODEL_MAP)
        self.assertIn("Wan2.2", VAE_MODEL_MAP)
        self.assertIn("LightVAE_W2.1", VAE_MODEL_MAP)
        self.assertIn("TAE_W2.2", VAE_MODEL_MAP)
        self.assertIn("LightTAE_HY1.5", VAE_MODEL_MAP)
        
        # Test each entry has required keys (updated for new schema)
        for key, value in VAE_MODEL_MAP.items():
            self.assertIn("class", value)
            self.assertIn("file", value)
            self.assertIn("internal_name", value)
            self.assertIn("url", value)  # New: Direct URL for auto-download
            self.assertIn("dim", value)  # New: VAE dimension
            self.assertIn("z_dim", value)  # New: z dimension
        
        # CRITICAL: Verify DISTINCT file paths (no reuse)
        files = [VAE_MODEL_MAP[k]["file"] for k in VAE_MODEL_MAP]
        self.assertEqual(len(files), len(set(files)), "VAE files must be DISTINCT - no reuse!")
        
        # Verify specific file mappings
        self.assertEqual(VAE_MODEL_MAP["Wan2.1"]["file"], "Wan2.1_VAE.pth")
        self.assertEqual(VAE_MODEL_MAP["Wan2.2"]["file"], "Wan2.2_VAE.pth")
        self.assertEqual(VAE_MODEL_MAP["LightVAE_W2.1"]["file"], "lightvaew2_1.pth")
        self.assertEqual(VAE_MODEL_MAP["TAE_W2.2"]["file"], "taew2_2.safetensors")
        self.assertEqual(VAE_MODEL_MAP["LightTAE_HY1.5"]["file"], "lighttaehy1_5.pth")

    def test_vae_model_in_node_input_types(self):
        """Test that vae_model parameter is present in node INPUT_TYPES."""
        init_types = FlashVSRNodeInitPipe.INPUT_TYPES()
        self.assertIn('vae_model', init_types['required'])
        # Ensure old parameters are removed
        self.assertNotIn('vae_type', init_types['required'])
        self.assertNotIn('alt_vae', init_types['required'])
        
        node_types = FlashVSRNode.INPUT_TYPES()
        self.assertIn('vae_model', node_types['required'])
        self.assertNotIn('vae_type', node_types['required'])

    # ==========================================================================
    # FIX 9: Tests for Pre-Flight Resource Calculator
    # ==========================================================================
    def test_estimate_vram_usage_modes(self):
        """Test estimate_vram_usage returns different values for different modes."""
        vram_full = estimate_vram_usage(1280, 720, 100, 2, mode='full')
        vram_tiny = estimate_vram_usage(1280, 720, 100, 2, mode='tiny')
        vram_tiny_long = estimate_vram_usage(1280, 720, 100, 2, mode='tiny-long')
        
        # All should return positive values
        self.assertGreater(vram_full, 0)
        self.assertGreater(vram_tiny, 0)
        self.assertGreater(vram_tiny_long, 0)
        
        # Full mode should use most VRAM
        self.assertGreater(vram_full, vram_tiny_long)
    
    def test_estimate_vram_usage_tiling_reduces(self):
        """Test that tiling reduces estimated VRAM."""
        vram_no_tile = estimate_vram_usage(1280, 720, 100, 2, tiled_vae=False, tiled_dit=False)
        vram_tiled = estimate_vram_usage(1280, 720, 100, 2, tiled_vae=True, tiled_dit=True)
        
        self.assertLess(vram_tiled, vram_no_tile)
    
    def test_estimate_vram_usage_chunking_reduces(self):
        """Test that chunking reduces estimated VRAM."""
        vram_no_chunk = estimate_vram_usage(1280, 720, 100, 2, chunk_size=0)
        vram_chunked = estimate_vram_usage(1280, 720, 100, 2, chunk_size=25)
        
        self.assertLess(vram_chunked, vram_no_chunk)
    
    def test_get_optimal_settings_high_vram(self):
        """Test that high VRAM returns default settings."""
        settings = get_optimal_settings(640, 480, 50, 2, available_vram_gb=32.0, mode='full')
        
        # With 32GB VRAM, should be safe with defaults
        self.assertFalse(settings['tiled_vae'])
        self.assertFalse(settings['tiled_dit'])
        self.assertEqual(settings['chunk_size'], 0)
        self.assertEqual(settings['resize_factor'], 1.0)
    
    def test_get_optimal_settings_low_vram(self):
        """Test that low VRAM suggests tiling and chunking."""
        settings = get_optimal_settings(1920, 1080, 200, 4, available_vram_gb=4.0, mode='full')
        
        # With only 4GB for 4K upscale, should recommend aggressive settings
        self.assertTrue(settings['tiled_vae'])
        self.assertTrue(settings['tiled_dit'])
        # Should recommend some chunking or resize
        self.assertTrue(settings['chunk_size'] > 0 or settings['resize_factor'] < 1.0)

    def test_vae_factory_function(self):
        """Test the create_video_vae factory function."""
        # Test wan2.1
        vae1 = create_video_vae('wan2.1')
        self.assertIsInstance(vae1, WanVideoVAE)
        
        # Test wan2.2
        vae2 = create_video_vae('wan2.2')
        self.assertIsInstance(vae2, Wan22VideoVAE)
        
        # Test lightx2v
        vae3 = create_video_vae('lightx2v', use_full_arch=True)
        self.assertIsInstance(vae3, LightX2VVAE)

    def test_wan22_vae_initialization(self):
        """Test Wan22VideoVAE initialization and basic attributes."""
        vae = Wan22VideoVAE(z_dim=16, dim=96)
        self.assertEqual(vae.vae_type, "wan2.2")
        self.assertEqual(vae.upsampling_factor, 8)
        self.assertIsNotNone(vae.model)

    def test_lightx2v_vae_initialization(self):
        """Test LightX2VVAE initialization and basic attributes."""
        vae = LightX2VVAE(z_dim=16, dim=64, use_full_arch=True)
        self.assertEqual(vae.vae_type, "lightx2v")
        self.assertEqual(vae.upsampling_factor, 8)
        self.assertIsNotNone(vae.model)

    def test_wan22_use_wan21_stats(self):
        """Test Wan22VideoVAE can switch to Wan2.1 statistics."""
        vae = Wan22VideoVAE(z_dim=16, dim=96)
        original_vae_type = vae.vae_type
        vae.use_wan21_stats()
        self.assertEqual(vae.vae_type, "wan2.1_compat")
        self.assertNotEqual(vae.vae_type, original_vae_type)

    def test_vae_factory_invalid_type(self):
        """Test that factory raises error for invalid VAE type."""
        with self.assertRaises(ValueError):
            create_video_vae('invalid_vae_type')

    def test_full_pipeline_vram_optimization(self):
        # Verify our changes to load_models_to_device usage in FlashVSRFullPipeline
        # We'll mock the internal methods
        pipe = FlashVSRFullPipeline(device="cpu")
        pipe.load_models_to_device = MagicMock()
        pipe.offload_model = MagicMock()
        pipe.decode_video = MagicMock(return_value=torch.zeros((1, 3, 5, 64, 64)))
        pipe.dit = MagicMock()
        pipe.vae = MagicMock()
        pipe.prompt_emb_posi = {'context': torch.zeros(1), 'stats': 'load'}
        pipe.generate_noise = MagicMock(return_value=torch.zeros((1, 16, 5, 8, 8)))

        # Mock global function model_fn_wan_video
        import src.pipelines.flashvsr_full
        src.pipelines.flashvsr_full.model_fn_wan_video = MagicMock(return_value=(torch.zeros((1, 16, 5, 8, 8)), None, None))

        # Run __call__
        try:
            pipe(
                prompt="test",
                num_frames=5,
                height=64,
                width=64,
                unload_dit=True,
                force_offload=True,
                enable_debug_logging=True
            )
        except Exception as e:
            # We expect it might fail due to tensor mismatches or other mocks,
            # but we want to check the call order of load_models_to_device
            print(f"Caught expected exception during mock run: {e}")
            pass

        # Check if load_models_to_device was called with ["dit"] first
        # We need to inspect the calls
        calls = pipe.load_models_to_device.call_args_list
        # print(calls)
        # Expected sequence:
        # 1. init_cross_kv -> load_models_to_device(["dit"]) -> load_models_to_device([])
        # 2. __call__ start -> load_models_to_device(["dit"]) (This is our CHANGE)
        # 3. offload_model(keep_vae=True) -> load_models_to_device(["vae"])
        # 4. offload_model() -> load_models_to_device([])

        # Verify that we see a call with ["dit"]
        found_dit_only = False
        for call in calls:
            args, _ = call
            if args[0] == ["dit"]:
                found_dit_only = True
                break
        self.assertTrue(found_dit_only, "Should have called load_models_to_device(['dit'])")

if __name__ == '__main__':
    unittest.main()
