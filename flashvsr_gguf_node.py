"""
FlashVSR GGUF Loader Node for ComfyUI
======================================
ComfyUI node for loading FlashVSR models in GGUF format with automatic 5D tensor reshaping.

This loader handles GGUF files that contain flattened 5D tensors (required due to GGUF's 4D limitation).
It automatically detects and reshapes tensors using metadata stored in the GGUF file.
"""

import os
import torch
import folder_paths

try:
    from .src.models.utils import load_state_dict
    from .src.models.model_manager import ModelManager
except ImportError:
    from src.models.utils import load_state_dict
    from src.models.model_manager import ModelManager


class FlashVSR_GGUF_Loader:
    """
    ComfyUI node for loading FlashVSR models from GGUF format.
    
    Automatically handles 5D tensors that were flattened during GGUF conversion,
    reshaping them back to their original dimensions using metadata stored in the file.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the ComfyUI node."""
        # FlashVSR model options
        model_options = ["FlashVSR", "FlashVSR-v1.1"]
        
        # Get list of GGUF files from FlashVSR model directories
        gguf_files = []
        seen_files = set()
        try:
            # Look in both FlashVSR and FlashVSR-v1.1 directories
            for model_name in model_options:
                model_path = os.path.join(folder_paths.models_dir, model_name)
                if os.path.exists(model_path):
                    for file in os.listdir(model_path):
                        if file.endswith(".gguf") and file not in seen_files:
                            gguf_files.append(file)
                            seen_files.add(file)
        except (OSError, AttributeError):
            pass
        
        if not gguf_files:
            gguf_files = ["No GGUF files found"]
        
        return {
            "required": {
                "model_name": (model_options, {
                    "default": "FlashVSR-v1.1"
                }),
                "gguf_file": (gguf_files, {
                    "default": gguf_files[0] if gguf_files else "No GGUF files found"
                }),
                "torch_dtype": (["auto", "float32", "float16", "bfloat16"], {
                    "default": "auto"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL_MANAGER",)
    RETURN_NAMES = ("model_manager",)
    FUNCTION = "load_gguf_model"
    CATEGORY = "FlashVSR"
    DESCRIPTION = "Load FlashVSR models from GGUF format with automatic 5D tensor reshaping"
    
    def load_gguf_model(self, model_name, gguf_file, torch_dtype="auto"):
        """
        Load a GGUF model file and create a ModelManager with the loaded models.
        
        Args:
            model_name: FlashVSR model directory name (FlashVSR or FlashVSR-v1.1)
            gguf_file: Name of the GGUF file to load
            torch_dtype: Target dtype for tensors (auto, float32, float16, bfloat16)
            
        Returns:
            Tuple containing the ModelManager with loaded models
        """
        print(f"\n{'='*60}")
        print(f"🚀 FlashVSR GGUF Engine Active: Loading {gguf_file}")
        print(f"{'='*60}")
        
        # Construct path to GGUF file in FlashVSR model directory
        model_path = os.path.join(folder_paths.models_dir, model_name)
        file_path = os.path.join(model_path, gguf_file)
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Fall back to checking other model directories
            alternate_models = ["FlashVSR", "FlashVSR-v1.1"]
            for alt_model in alternate_models:
                if alt_model != model_name:
                    alt_path = os.path.join(folder_paths.models_dir, alt_model, gguf_file)
                    if os.path.exists(alt_path):
                        file_path = alt_path
                        print(f"⚠️ Found GGUF file in {alt_model} instead of {model_name}")
                        break
            else:
                raise FileNotFoundError(
                    f"GGUF file not found: {gguf_file}\n"
                    f"Expected location: {file_path}\n"
                    f"Please place GGUF files in: {model_path}"
                )
        
        print(f"📂 Model directory: {model_path}")
        print(f"📄 Loading GGUF file: {file_path}")
        
        # Determine torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.float16,  # Default to float16 for memory efficiency
        }
        
        target_dtype = dtype_map.get(torch_dtype, torch.float16)
        print(f"Target dtype: {target_dtype}")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Target device: {device}")
        
        # Load state dict from GGUF
        try:
            state_dict = load_state_dict(file_path, torch_dtype=target_dtype)
            print(f"Successfully loaded {len(state_dict)} tensors from GGUF file")
            
            # Log any 5D tensors that were reshaped
            reshaped_count = 0
            for name, tensor in state_dict.items():
                if len(tensor.shape) == 5:
                    reshaped_count += 1
                    print(f"  5D tensor: {name} -> {tensor.shape}")
            
            if reshaped_count > 0:
                print(f"Total 5D tensors reshaped: {reshaped_count}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GGUF file: {file_path}\n"
                f"Error: {str(e)}"
            )
        
        # Create ModelManager and load the model
        try:
            model_manager = ModelManager(
                torch_dtype=target_dtype,
                device=device,
                file_path_list=[]
            )
            
            # Load the model using the state dict
            model_manager.load_model(file_path, device=device, torch_dtype=target_dtype)
            
            print(f"Model loaded successfully")
            print(f"Loaded models: {model_manager.model_name}")
            print(f"{'='*60}\n")
            
            return (model_manager,)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to create model from GGUF state dict\n"
                f"Error: {str(e)}\n"
                f"Make sure the GGUF file contains a valid FlashVSR model"
            )


# ComfyUI node registration mapping
NODE_CLASS_MAPPINGS = {
    "FlashVSR_GGUF_Loader": FlashVSR_GGUF_Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSR_GGUF_Loader": "FlashVSR GGUF Loader",
}
