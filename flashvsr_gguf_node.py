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
        # Get list of GGUF files from checkpoints folder
        gguf_files = []
        seen_files = set()
        try:
            checkpoints_dir = folder_paths.get_folder_paths("checkpoints")
            for checkpoint_dir in checkpoints_dir:
                if os.path.exists(checkpoint_dir):
                    for file in os.listdir(checkpoint_dir):
                        if file.endswith(".gguf") and file not in seen_files:
                            gguf_files.append(file)
                            seen_files.add(file)
        except (OSError, AttributeError):
            pass
        
        if not gguf_files:
            gguf_files = ["No GGUF files found"]
        
        return {
            "required": {
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
    
    def load_gguf_model(self, gguf_file, torch_dtype="auto"):
        """
        Load a GGUF model file and create a ModelManager with the loaded models.
        
        Args:
            gguf_file: Name of the GGUF file to load
            torch_dtype: Target dtype for tensors (auto, float32, float16, bfloat16)
            
        Returns:
            Tuple containing the ModelManager with loaded models
        """
        print(f"\n{'='*60}")
        print(f"FlashVSR GGUF Loader")
        print(f"{'='*60}")
        
        # Find the full path to the GGUF file
        file_path = None
        checkpoints_dirs = folder_paths.get_folder_paths("checkpoints")
        
        for checkpoint_dir in checkpoints_dirs:
            potential_path = os.path.join(checkpoint_dir, gguf_file)
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        
        if file_path is None:
            raise FileNotFoundError(
                f"GGUF file not found: {gguf_file}\n"
                f"Searched in: {checkpoints_dirs}"
            )
        
        print(f"Loading GGUF file: {file_path}")
        
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
