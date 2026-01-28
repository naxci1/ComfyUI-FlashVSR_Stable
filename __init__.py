from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Import GGUF loader node
try:
    from .flashvsr_gguf_node import NODE_CLASS_MAPPINGS as GGUF_NODE_CLASS_MAPPINGS
    from .flashvsr_gguf_node import NODE_DISPLAY_NAME_MAPPINGS as GGUF_NODE_DISPLAY_NAME_MAPPINGS
    
    # Merge GGUF node mappings with existing mappings
    NODE_CLASS_MAPPINGS.update(GGUF_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(GGUF_NODE_DISPLAY_NAME_MAPPINGS)
except ImportError as e:
    print(f"Warning: Could not import GGUF loader node: {e}")
    print("Make sure 'gguf' library is installed: pip install gguf")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
