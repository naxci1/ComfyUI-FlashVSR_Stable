"""
Tensor Utilities for GGUF Model Loading
=========================================
Utilities for handling 5D tensors in GGUF format that have been flattened due to GGUF's 4D limitation.
This module provides functions to detect and reshape flattened tensors back to their original dimensions.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def has_raw_shape_metadata(tensor_metadata: dict) -> bool:
    """
    Check if a tensor has raw_shape metadata indicating it was originally 5D.
    
    Args:
        tensor_metadata: Dictionary containing tensor metadata from GGUF file
        
    Returns:
        True if tensor has raw_shape metadata with 5 dimensions, False otherwise
    """
    if not isinstance(tensor_metadata, dict):
        return False
    
    raw_shape = tensor_metadata.get('raw_shape', None)
    if raw_shape is None:
        return False
    
    # Check if raw_shape indicates 5D tensor
    if isinstance(raw_shape, (list, tuple)) and len(raw_shape) == 5:
        return True
    
    return False


def reshape_flattened_tensor(tensor: torch.Tensor, raw_shape: Tuple[int, ...], 
                             tensor_name: str = "unknown") -> torch.Tensor:
    """
    Reshape a flattened tensor back to its original 5D dimensions.
    
    Args:
        tensor: The flattened tensor (typically 1D or 2D)
        raw_shape: The original 5D shape tuple (e.g., [1536, 16, 1, 2, 2])
        tensor_name: Name of the tensor for error reporting
        
    Returns:
        Reshaped tensor with original 5D dimensions
        
    Raises:
        ValueError: If tensor size doesn't match the product of raw_shape dimensions
    """
    if not isinstance(raw_shape, (list, tuple)):
        raise ValueError(f"raw_shape must be a list or tuple, got {type(raw_shape)}")
    
    # Calculate expected total elements
    expected_size = 1
    for dim in raw_shape:
        expected_size *= dim
    
    # Get actual tensor size
    actual_size = tensor.numel()
    
    # Validate sizes match
    if actual_size != expected_size:
        raise ValueError(
            f"Tensor '{tensor_name}' size mismatch: "
            f"actual size={actual_size}, expected size from raw_shape={expected_size}. "
            f"raw_shape={raw_shape}, current_shape={tensor.shape}"
        )
    
    # Reshape tensor to original 5D dimensions
    try:
        reshaped_tensor = tensor.reshape(raw_shape)
        return reshaped_tensor
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to reshape tensor '{tensor_name}' to {raw_shape}: {str(e)}"
        )


def process_gguf_tensor(tensor: torch.Tensor, tensor_metadata: dict, 
                        tensor_name: str = "unknown") -> torch.Tensor:
    """
    Process a GGUF tensor, reshaping it if needed based on metadata.
    
    This is the main entry point for processing GGUF tensors. It checks for raw_shape
    metadata and automatically reshapes 5D tensors that were flattened.
    
    Args:
        tensor: The tensor from GGUF file
        tensor_metadata: Metadata dictionary for this tensor
        tensor_name: Name of the tensor for logging and error reporting
        
    Returns:
        Processed tensor (reshaped if necessary, unchanged otherwise)
    """
    # Check if this tensor needs reshaping
    if has_raw_shape_metadata(tensor_metadata):
        raw_shape = tensor_metadata['raw_shape']
        
        # Only reshape if raw_shape indicates 5D
        if len(raw_shape) == 5:
            print(f"  Reshaping tensor '{tensor_name}' from {tensor.shape} to {raw_shape}")
            return reshape_flattened_tensor(tensor, raw_shape, tensor_name)
    
    # No reshaping needed
    return tensor


def validate_5d_tensor_shape(shape: Tuple[int, ...], tensor_name: str = "unknown") -> bool:
    """
    Validate that a 5D tensor shape is reasonable for FlashVSR architecture.
    
    Args:
        shape: The tensor shape to validate
        tensor_name: Name of the tensor for error reporting
        
    Returns:
        True if shape is valid
        
    Raises:
        ValueError: If shape is invalid
    """
    if len(shape) != 5:
        raise ValueError(f"Expected 5D tensor shape, got {len(shape)}D for '{tensor_name}'")
    
    # Basic sanity checks for FlashVSR 3D conv layers
    # Typical shapes should be [out_channels, in_channels, depth, height, width]
    for i, dim in enumerate(shape):
        if dim <= 0:
            raise ValueError(
                f"Invalid dimension {i} in shape {shape} for '{tensor_name}': "
                f"dimension must be positive, got {dim}"
            )
    
    # Check reasonable bounds (prevent accidentally loading corrupted data)
    max_channels = 100000  # Reasonable upper bound for channel dimensions
    max_spatial = 1000     # Reasonable upper bound for spatial dimensions
    
    if shape[0] > max_channels or shape[1] > max_channels:
        raise ValueError(
            f"Channel dimensions too large in shape {shape} for '{tensor_name}': "
            f"channels exceed {max_channels}"
        )
    
    if shape[2] > max_spatial or shape[3] > max_spatial or shape[4] > max_spatial:
        raise ValueError(
            f"Spatial dimensions too large in shape {shape} for '{tensor_name}': "
            f"spatial dims exceed {max_spatial}"
        )
    
    return True
